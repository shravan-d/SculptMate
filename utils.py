def label_multiline(layout, text='', icon='NONE', width=-1, max_lines=12, use_urls=True, alignment="LEFT", alert=False):
    '''
     draw a ui label, but try to split it in multiple lines.

    Parameters
    ----------
    layout
    text
    icon
    width width to split by in px
    max_lines maximum lines to draw
    use_urls - automatically parse urls to buttons
    Returns
    -------
    rows of the text(to add extra elements)
    '''
    rows = []
    if text.strip() == '':
        return [layout.row()]

    text = text.replace("\r\n", "\n")

    lines = text.split("\n")

    if width > 0:
        char_threshold = int(width / 5.7)
    else:
        char_threshold = 35

    line_index = 0
    for line in lines:

        line_index += 1
        while len(line) > char_threshold:
            #find line split close to the end of line
            i = line.rfind(" ", 0, char_threshold)
            #split long words
            if i < 1:
                i = char_threshold
            l1 = line[:i]

            row = layout.row()
            if alert: row.alert = True
            row.alignment = alignment
            row.label(text=l1, icon=icon)
            rows.append(row)

            # set the icon to none after the first row
            icon = "NONE"

            line = line[i:].lstrip()
            line_index += 1
            if line_index > max_lines:
                break

        if line_index > max_lines:
            break

        row = layout.row()
        if alert: row.alert = True
        row.alignment = alignment
        row.label(text=line, icon=icon)
        rows.append(row)

        # set the icon to none after the first row
        icon = "NONE"

    # return the resulting rows
    return rows