import webview
    
def open_in_browser(url):
    webview.create_window("Gesturesphone", url, resizable=True)
    webview.start()
    return False
