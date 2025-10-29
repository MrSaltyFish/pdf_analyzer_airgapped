import MessageBubble from "./MessageBubble";

export default function ChatWindow({
  messages,
}: {
  messages: { role: string; content: string }[];
}) {
  return (
    <div className="flex-1 overflow-y-auto border border-gray-300 rounded-lg p-3 bg-white mb-4">
      {messages.length === 0 && (
        <p className="text-gray-400 text-center mt-10">
          Upload a PDF and start chatting!
        </p>
      )}
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} role={msg.role} content={msg.content} />
      ))}
    </div>
  );
}
