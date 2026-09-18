import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                Text {
                    text: "Red Team & Failures"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }
                Text {
                    text: "Seeded adversarial runs · immutable failures · regression promotion"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            SpinBox {
                id: seedBox
                from: 0
                to: 2147483647
                value: 1337
                editable: true
                enabled: !appController.redTeamFailures.busy
                Accessible.name: "Red-team seed"
            }

            SpinBox {
                id: caseLimitBox
                from: 1
                to: 5000
                value: 100
                editable: true
                enabled: !appController.redTeamFailures.busy
                Accessible.name: "Maximum base cases"
            }

            LabButton {
                text: appController.redTeamFailures.busy ? "Running…" : "Run red team"
                primary: true
                enabled: appController.redTeamFailures.canRunRedTeam
                onClicked: appController.redTeamFailures.runRedTeam(seedBox.value, caseLimitBox.value)
            }

            LabButton {
                text: "Cancel"
                visible: appController.redTeamFailures.busy
                enabled: appController.redTeamFailures.busy
                onClicked: appController.redTeamFailures.cancelRedTeam()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: runnerMessage.implicitHeight + 20
            radius: 8
            color: Theme.surfaceAlt
            border.color: Theme.border

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 10
                BusyIndicator {
                    visible: appController.redTeamFailures.busy
                    running: visible
                    implicitWidth: 20
                    implicitHeight: 20
                }
                Text {
                    id: runnerMessage
                    Layout.fillWidth: true
                    text: appController.redTeamFailures.runnerMessage
                    color: Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Repeater {
                model: [
                    {label: "FAILURES", value: appController.redTeamFailures.failureTotal, tone: "neutral"},
                    {label: "VETO", value: appController.redTeamFailures.vetoFailureTotal, tone: appController.redTeamFailures.vetoFailureTotal > 0 ? "bad" : "neutral"},
                    {label: "REGRESSIONS", value: appController.redTeamFailures.regressionTotal, tone: "neutral"},
                    {label: "RUNS", value: appController.redTeamFailures.runTotal, tone: "neutral"}
                ]
                delegate: Panel {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: 66
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 1
                        Text { text: modelData.label; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                        Text {
                            text: modelData.value
                            color: modelData.tone === "bad" ? Theme.badText : Theme.text
                            font.pixelSize: 20
                            font.weight: Font.DemiBold
                        }
                    }
                }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Panel {
                SplitView.preferredWidth: 300
                SplitView.minimumWidth: 230
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8
                    Text { text: "COMPLETED EXPERIMENTS"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    ListView {
                        id: experimentList
                        Layout.fillWidth: true
                        Layout.preferredHeight: 145
                        clip: true
                        spacing: 3
                        model: appController.redTeamFailures.experimentRows
                        delegate: ItemDelegate {
                            required property var modelData
                            width: experimentList.width
                            height: 48
                            activeFocusOnTab: true
                            Accessible.name: "Red-team experiment " + modelData.name
                            onClicked: appController.redTeamFailures.selectExperiment(modelData.id)
                            contentItem: ColumnLayout {
                                spacing: 1
                                Text { Layout.fillWidth: true; text: modelData.name; color: Theme.text; font.pixelSize: 11; elide: Text.ElideRight }
                                Text { Layout.fillWidth: true; text: modelData.trainer; color: Theme.dim; font.pixelSize: 9; elide: Text.ElideMiddle }
                            }
                            background: Rectangle {
                                radius: 6
                                color: appController.redTeamFailures.selectedExperimentId === modelData.id ? Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent : "transparent"
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "RUN HISTORY"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                        Item { Layout.fillWidth: true }
                        Text { text: appController.redTeamFailures.runTotal + " total"; color: Theme.muted; font.pixelSize: 10 }
                    }
                    ListView {
                        id: runList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 3
                        model: appController.redTeamFailures.runRows
                        delegate: ItemDelegate {
                            required property var modelData
                            width: runList.width
                            height: 52
                            activeFocusOnTab: true
                            Accessible.name: "Red-team run " + modelData.id
                            onClicked: appController.redTeamFailures.selectRun(modelData.id)
                            contentItem: ColumnLayout {
                                spacing: 1
                                Text { Layout.fillWidth: true; text: modelData.suite; color: Theme.text; font.pixelSize: 10; elide: Text.ElideRight }
                                Text { text: modelData.status + " · seed " + modelData.seed; color: Theme.dim; font.pixelSize: 9 }
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        LabButton { text: "‹"; enabled: appController.redTeamFailures.canPreviousRunPage; onClicked: appController.redTeamFailures.previousRunPage() }
                        Text { text: "Page " + appController.redTeamFailures.runPageNumber; color: Theme.muted; font.pixelSize: 10 }
                        Item { Layout.fillWidth: true }
                        LabButton { text: "›"; enabled: appController.redTeamFailures.canNextRunPage; onClicked: appController.redTeamFailures.nextRunPage() }
                    }
                }
            }

            Panel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 430
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "FAILURE LIBRARY"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                        Item { Layout.fillWidth: true }
                        ComboBox {
                            id: severityFilter
                            model: ["ALL", "VETO", "NON_VETO"]
                            currentIndex: Math.max(0, model.indexOf(appController.redTeamFailures.failureSeverity))
                            onActivated: appController.redTeamFailures.setFailureSeverity(currentText)
                            Accessible.name: "Failure severity filter"
                        }
                        Text { text: appController.redTeamFailures.failureTotal + " failures"; color: Theme.muted; font.pixelSize: 10 }
                    }

                    ListView {
                        id: failureList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 2
                        model: appController.redTeamFailures.failureRows
                        delegate: ItemDelegate {
                            required property var modelData
                            width: failureList.width
                            height: 46
                            activeFocusOnTab: true
                            Accessible.name: "Failure " + modelData.kind
                            onClicked: appController.redTeamFailures.selectFailure(modelData.id)
                            contentItem: RowLayout {
                                spacing: 8
                                StatusPill { text: modelData.severity; tone: modelData.severity === "VETO" ? "bad" : "warn" }
                                Text { Layout.preferredWidth: 170; text: modelData.kind; color: Theme.text; font.pixelSize: 10; elide: Text.ElideRight }
                                Text { Layout.preferredWidth: 70; text: modelData.split || "—"; color: Theme.dim; font.pixelSize: 9 }
                                Text { Layout.fillWidth: true; text: modelData.exampleId || modelData.id; color: Theme.muted; font.pixelSize: 9; elide: Text.ElideMiddle }
                            }
                        }
                        Text {
                            anchors.centerIn: parent
                            visible: failureList.count === 0
                            text: "No failures recorded for this filter."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        LabButton { text: "‹"; enabled: appController.redTeamFailures.canPreviousFailurePage; onClicked: appController.redTeamFailures.previousFailurePage() }
                        Text { text: "Page " + appController.redTeamFailures.failurePageNumber; color: Theme.muted; font.pixelSize: 10 }
                        Item { Layout.fillWidth: true }
                        LabButton { text: "›"; enabled: appController.redTeamFailures.canNextFailurePage; onClicked: appController.redTeamFailures.nextFailurePage() }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        visible: Object.keys(appController.redTeamFailures.selectedFailure).length > 0
                        implicitHeight: detailColumn.implicitHeight + 20
                        radius: 8
                        color: Theme.surfaceAlt
                        border.color: Theme.border
                        ColumnLayout {
                            id: detailColumn
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 10
                            spacing: 5
                            RowLayout {
                                Layout.fillWidth: true
                                Text { Layout.fillWidth: true; text: appController.redTeamFailures.selectedFailure.kind || ""; color: Theme.text; font.pixelSize: 12; font.weight: Font.DemiBold }
                                StatusPill {
                                    text: appController.redTeamFailures.selectedFailure.isRegression ? "REGRESSION" : "OPEN"
                                    tone: appController.redTeamFailures.selectedFailure.isRegression ? "good" : "neutral"
                                }
                                LabButton {
                                    text: "Promote to regression"
                                    enabled: !appController.redTeamFailures.selectedFailure.isRegression
                                    onClicked: appController.redTeamFailures.promoteSelectedFailure("default")
                                }
                            }
                            Text { Layout.fillWidth: true; text: "Expected\n" + (appController.redTeamFailures.selectedFailure.expected || ""); color: Theme.muted; font.family: "Consolas"; font.pixelSize: 9; wrapMode: Text.WrapAnywhere }
                            Text { Layout.fillWidth: true; text: "Observed\n" + (appController.redTeamFailures.selectedFailure.observed || ""); color: Theme.muted; font.family: "Consolas"; font.pixelSize: 9; wrapMode: Text.WrapAnywhere }
                        }
                    }
                }
            }
        }
    }
}
