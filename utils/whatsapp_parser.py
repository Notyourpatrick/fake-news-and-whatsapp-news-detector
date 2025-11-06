def extract_messages_from_whatsapp(file_content):
    lines = file_content.decode("utf-8").split("\n")
    messages = []
    for line in lines:
        if " - " in line:
            parts = line.split(" - ", 1)
            if len(parts) > 1 and ": " in parts[1]:
                msg = parts[1].split(": ", 1)[1]
                messages.append(msg)
    return messages
