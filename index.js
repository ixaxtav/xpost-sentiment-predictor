const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.question("Enter your text: ", (input) => {
  require("child_process").exec(
    `node scripts/analyze.js "${input}"`,
    (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${stderr}`);
      } else {
        console.log(stdout);
      }
      rl.close();
    }
  );
});
