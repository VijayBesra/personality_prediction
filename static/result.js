const Username = ""
function knowMore(personalityType, gotText) {
    fewTraits = parseForSentence(gotText)
    let mainContent = document.getElementById("mainContent");

    const greeting = document.createElement("p")
    greeting.innerText = "Hello " + Username
    greeting.className = "greeting-p"
    mainContent.appendChild(greeting)

    let traitsArray = []

    fewTraits.forEach(element => {
        let ele = document.createElement("span")
        ele.innerText = element[0]

        ele.className = "trait-span"
        traitsArray.push(ele)
    });

    const introductoryTextElement = document.createElement("div")
    introductoryTextElement.innerHTML = "Your personality type is " + personalityType + ", and keeping this information at the back of your mind will be crucial for helping you learn more about yourself. Being "

    introductoryTextElement.appendChild(traitsArray[0])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[1])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[2])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[3])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[4])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[5])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[6])
    introductoryTextElement.append(", ")
    introductoryTextElement.appendChild(traitsArray[7])
    introductoryTextElement.append(" and ")
    introductoryTextElement.appendChild(traitsArray[8])

    introductoryTextElement.append(" are some of your most notable traits.")

    introductoryTextElement.className = "introductory-div"

    // " and " + traitsArray[8] + ", " + " are some of your most notable traits."


    mainContent.appendChild(introductoryTextElement)
    console.log(greeting)
}


function parseForSentence(gotText) {
    const jsonData = JSON.parse(gotText)[0]
    let sorted = []

    for (var trait in jsonData) {
        sorted.push([trait, jsonData[trait]])
    }

    sorted.sort(function (traitA, traitB) {
        return traitB[1] - traitA[1];
    })

    return sorted.slice(2, 11)
}