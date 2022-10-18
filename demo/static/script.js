// send user query to get bot response
$('#user_query').keypress(function(e) { 
    if (e.keyCode === 13) {
        e.preventDefault();

        // get user query and display to dialog history
        var user_query = $('#user_query').val().trim();
        if (user_query.length == 0) {
            alert('No message is typed.')
            return;
        }
        var dialog_history = document.querySelector(".dialog_history");
        dialog_history.innerHTML += `<p class="user_msg">${user_query}</p>`;
        $('.dialog_history').scrollTop($('.dialog_history')[0].scrollHeight);
        $('#user_query').val('');

        // get context from dialog history
        var context = [];
        var context_list = document.querySelectorAll('[class*="msg"]');
        for (var i = 0; i < context_list.length; i++) {
            context.push(context_list[i].innerText);
        }
        
        // call endpoint
        $.ajax({
            url: '/send',
            method: 'post',
            data: JSON.stringify({context: context}),
            dataType: 'json',
            contentType: 'application/json',
            success: function(res_data) {
                var dialog_history = document.querySelector(".dialog_history");
                dialog_history.innerHTML += `<p class="bot_msg">${res_data.bot_response}</p>`;
                dialog_history.innerHTML += `<img class=image src=${res_data.image_url}>`;
                $('.dialog_history').scrollTop($('.dialog_history')[0].scrollHeight);
            }
        });
    }
});