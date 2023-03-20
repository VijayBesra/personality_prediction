function checkInput(){
    const input = document.getElementById('inputText').value
    console.log(input)
    if(!is_correct_Sentence(input)){
        toggleErrors()
        return false
    }
    return true
}

function toggleErrors(){
    document.getElementsByClassName('requiredFromYou')[0].style.visibility='visible'
}

function is_correct_Sentence(input_str) {
    const first_char = input_str[0];
    const last_char = input_str[input_str.length - 1];
    return /[a-zA-Z]/.test(first_char) && last_char == "."
}

