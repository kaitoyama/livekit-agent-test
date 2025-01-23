import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    transcription,
)
from livekit.plugins import azure, silero, openai,google

load_dotenv()

logger = logging.getLogger("transcriber")


async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # you may not want to log interim transcripts, they are not final and may be incorrect
            pass
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print(" -> ", ev.alternatives[0].text)
        elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
            logger.debug(f"metrics: {ev.recognition_usage}")

        stt_forwarder.update(ev)


async def entrypoint(ctx: JobContext):
    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")
    # this example uses OpenAI Whisper, but you can use assemblyai, deepgram, google, azure, etc.
    stt_impl = azure.STT(language="ja-JP")

    if not stt_impl.capabilities.streaming:
        # wrap with a stream adapter to use streaming semantics
        stt_impl = stt.StreamAdapter(
            stt=stt_impl,
            vad=silero.VAD.load(
                min_silence_duration=0.2,
            ),
        )

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, 
            participant=participant, 
            track=track
        )

        stt_stream = stt_impl.stream()
        
        # 各参加者の音声を個別に処理
        async def forward_user_transcription():
            async for ev in stt_stream:
                if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    pass
                elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    # 参加者IDと一緒に文字起こし結果を出力
                    print(f"[{participant.identity}] -> {ev.alternatives[0].text}")
                elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    logger.debug(f"metrics for {participant.identity}: {ev.recognition_usage}")

                stt_forwarder.update(ev)

        asyncio.create_task(forward_user_transcription())

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))