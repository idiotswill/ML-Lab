import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"
import "views"

ApplicationWindow {
    id: window

    visible: true
    width: 1280
    height: 720
    minimumWidth: 960
    minimumHeight: 640
    title: "Frankenhomie ML Lab"
    color: Theme.bg

    property int currentPage: 0
    property string toastText: ""
    property string errorTitle: ""
    property string errorText: ""

    palette.window: Theme.bg
    palette.windowText: Theme.text
    palette.base: Theme.surfaceAlt
    palette.text: Theme.text
    palette.button: Theme.surfaceAlt
    palette.buttonText: Theme.text2
    palette.highlight: Theme.accent
    palette.highlightedText: Theme.accentText
    palette.placeholderText: Theme.dim

    Shortcut {
        sequence: "Ctrl+1"
        enabled: appController.hasWorkspace
        onActivated: window.currentPage = 0
    }

    Shortcut {
        sequence: "Ctrl+2"
        enabled: appController.hasWorkspace
        onActivated: window.currentPage = 1
    }

    Shortcut {
        sequence: "Ctrl+3"
        enabled: appController.hasWorkspace
        onActivated: window.currentPage = 2
    }

    Shortcut {
        sequence: "Ctrl+4"
        enabled: appController.hasWorkspace
        onActivated: window.currentPage = 3
    }

    Connections {
        target: appController

        function onNoticeRaised(message) {
            window.toastText = message
            toast.visible = true
            toastTimer.restart()
        }

        function onErrorRaised(title, message) {
            window.errorTitle = title
            window.errorText = message
            errorDialog.open()
        }

        function onWorkspaceChanged() {
            window.currentPage = 0
        }
    }

    Timer {
        id: toastTimer
        interval: 3000
        onTriggered: toast.visible = false
    }

    Rectangle {
        id: toast
        visible: false
        z: 99
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 24
        radius: 9
        color: Theme.goodSurface
        border.color: Theme.goodBorder
        width: toastLabel.implicitWidth + 34
        height: 42

        Text {
            id: toastLabel
            anchors.centerIn: parent
            text: window.toastText
            color: Theme.goodText
            font.pixelSize: 13
        }
    }

    Dialog {
        id: errorDialog
        modal: true
        anchors.centerIn: parent
        width: 520
        title: window.errorTitle
        standardButtons: Dialog.Ok

        background: Rectangle {
            color: Theme.surface
            radius: 12
            border.color: Theme.badBorder
        }

        contentItem: Text {
            text: window.errorText
            color: Theme.badText
            wrapMode: Text.WordWrap
            width: 460
        }
    }

    Loader {
        anchors.fill: parent
        sourceComponent: appController.hasWorkspace ? shell : welcome
    }

    Component {
        id: welcome
        WelcomeView {}
    }

    Component {
        id: shell

        Item {
            RowLayout {
                anchors.fill: parent
                spacing: 0

                Rectangle {
                    Layout.preferredWidth: 226
                    Layout.fillHeight: true
                    color: Theme.sidebar
                    border.color: Theme.border

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8

                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.bottomMargin: 18

                            Text {
                                text: "ML LAB"
                                color: Theme.accent
                                font.pixelSize: 12
                                font.letterSpacing: 1.7
                                font.weight: Font.Bold
                            }

                            Text {
                                text: "Frankenhomie"
                                color: Theme.text
                                font.pixelSize: 19
                                font.weight: Font.DemiBold
                            }

                            Text {
                                text: appController.workspacePath
                                color: Theme.dim
                                font.pixelSize: 10
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                        }

                        Repeater {
                            model: ["Projects", "Jobs", "Diagnostics", "Settings"]

                            delegate: Button {
                                required property string modelData
                                required property int index

                                Layout.fillWidth: true
                                implicitHeight: 42
                                text: modelData
                                checkable: true
                                checked: window.currentPage === index
                                activeFocusOnTab: true
                                Accessible.role: Accessible.Button
                                Accessible.name: modelData
                                onClicked: window.currentPage = index

                                contentItem: Text {
                                    text: parent.text
                                    color: parent.checked ? Theme.text : Theme.muted
                                    font.pixelSize: 13
                                    font.weight: parent.checked ? Font.DemiBold : Font.Normal
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: 11
                                }

                                background: Rectangle {
                                    radius: 8
                                    color: parent.checked ? Theme.accentSurface
                                                          : (parent.hovered ? Theme.hover : "transparent")
                                    border.color: parent.activeFocus ? Theme.accent
                                                                    : (parent.checked ? Theme.accentBorder
                                                                                      : "transparent")
                                    border.width: parent.activeFocus ? 2 : 1
                                }
                            }
                        }

                        Item {
                            Layout.fillHeight: true
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.border
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            StatusPill {
                                text: "INTEGRATION NO-GO"
                                tone: "warn"
                            }

                            Item {
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    StackLayout {
                        anchors.fill: parent
                        anchors.margins: 24
                        currentIndex: window.currentPage

                        ProjectsView {}
                        JobsView {}
                        DiagnosticsView {}
                        SettingsView {}
                    }
                }
            }
        }
    }
}
