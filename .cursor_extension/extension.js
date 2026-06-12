const vscode = require('vscode');
const cp = require('child_process');
const path = require('path');

function activate(context) {
    console.log('HiPert Guardrail extension active.');

    let disposable = vscode.commands.registerCommand('hipert-guardrail.sanitizeSelection', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('HiPert: No active editor found.');
            return;
        }

        const selection = editor.selection;
        const text = editor.document.getText(selection);
        if (!text || text.trim() === '') {
            vscode.window.showWarningMessage('HiPert: No text selected.');
            return;
        }

        const langId = editor.document.languageId;
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            vscode.window.showErrorMessage('HiPert: Open a workspace folder first.');
            return;
        }

        const projectRoot = workspaceFolders[0].uri.fsPath;
        // Correctly target the script under main_code/deploy/
        const scriptPath = path.join(projectRoot, 'main_code', 'deploy', 'sanitize_api.py');

        const pythonProcess = cp.spawn('python', [scriptPath, '--lang', langId]);

        let outputData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            outputData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`HiPert error: ${errorData}`);
                vscode.window.showErrorMessage(`HiPert Failure: ${errorData}`);
                return;
            }

            editor.edit(editBuilder => {
                editBuilder.replace(selection, outputData);
            }).then(success => {
                if (success) {
                    vscode.window.showInformationMessage('HiPert: Code sanitized.');
                } else {
                    vscode.window.showErrorMessage('HiPert: Failed to apply text modification.');
                }
            });
        });

        pythonProcess.stdin.write(text);
        pythonProcess.stdin.end();
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};