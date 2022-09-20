// enter to send message
$('#message').keypress(function(e) { 
    if (e.keyCode === 13) {
        e.preventDefault();
        $("#send_btn").click();
    }
});


// send user message to get AI response
$('#send_btn').click(function(e) {
    // get input values
    e.preventDefault(); 
    var context = $('#context').val().trim();
    var message = $('#message').val().trim();
    if (message.length == 0) {
        alert('No message is typed.')
        return;
    }
    var button = $(this);
    button.prop('disabled', true);
    
    // send request
    $.ajax({
        url: '/send',
        method: 'post',
        data: JSON.stringify({
            context: context, 
            message: message
        }),
        dataType: 'json',
        contentType: 'application/json',
        success: function(response) {
            $('#result_div').removeClass('d-none');
            $('#context').val(response.context);
            $('#context').scrollTop($('#context')[0].scrollHeight);
            $('#message').val('');
            button.prop('disabled', false);
        }
    });
});