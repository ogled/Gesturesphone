function send(cmd) {
    fetch(`/api/command?cmd=${cmd}`)
        .then(r => r.json())
        .then(data => {
            document.getElementById("status").innerText =
                "Команда: " + data.cmd;
        });
}