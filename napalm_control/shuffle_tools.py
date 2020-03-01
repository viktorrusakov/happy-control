import sympy as sym


def shuffle_product(x, y):
    """
    Вычисление шафл-произведения двух элементов.
    """
    if '.' not in x:
        if '.' not in y:
            return x + '.' + y + '|' + y + '.' + x
        else:
            y_split = y.split('.')
            res_1 = x + '.' + y
            res_2 = '|'.join(map(lambda z: y_split[0] + '.' + z, shuffle_product('.'.join(y_split[1:]), x).split('|')))
            return res_1 + '|' + res_2
    elif '.' not in y:
        x_split = x.split('.')
        res_1 = '|'.join(map(lambda z: x_split[0] + '.' + z, shuffle_product('.'.join(x_split[1:]), y).split('|')))
        res_2 = y + '.' + x
        return res_1 + '|' + res_2
    else:
        x_split = x.split('.')
        y_split = y.split('.')
        res_1 = '|'.join(map(lambda z: x_split[0] + '.' + z, shuffle_product('.'.join(x_split[1:]), y).split('|')))
        res_2 = '|'.join(map(lambda z: y_split[0] + '.' + z, shuffle_product('.'.join(y_split[1:]), x).split('|')))
        return res_1 + '|' + res_2


def calc_shuffle_lin_comb(x='', y='', x_count=0, y_count=0):
    """
        Шафл-произведение линейной комбинации моментов. x_count, y_count - количество раз соответствующий
        элемент попадет в произведение.
    """
    if not x and y_count == 1:
        return y
    elif not y and x_count == 1:
        return x
    elif x_count == 0:
        if y_count == 0:
            return ''
        elif y_count == 1:
            return y
    elif y_count == 0 and x_count == 1:
        return x
    x_times = x_count - 1
    y_times = y_count - 1
    if x and x_times >= 0:
        x_split = x.split('|')
        res = [x]
        times = 1
        while times <= x_times:
            new_res = []
            for element in res:
                temp = element.split('|')
                for moment_1_repr in temp:
                    if 'x' not in moment_1_repr:
                        if moment_1_repr.startswith('-'):
                            moment_1, coef_1 = moment_1_repr[1:], sym.Rational('-1')
                        else:
                            moment_1, coef_1 = moment_1_repr, sym.Rational('1')
                    else:
                        split_moment_1 = moment_1_repr.split('x')
                        moment_1, coef_1 = split_moment_1[1], sym.Rational(split_moment_1[0])
                    for moment_2_repr in x_split:
                        if 'x' not in moment_2_repr:
                            if moment_2_repr.startswith('-'):
                                moment_2, coef_2 = moment_2_repr[1:], sym.Rational('-1')
                            else:
                                moment_2, coef_2 = moment_2_repr, sym.Rational('1')
                        else:
                            split_moment_2 = moment_2_repr.split('x')
                            moment_2, coef_2 = split_moment_2[1], sym.Rational(split_moment_2[0])
                        coef = str(coef_1 * coef_2)
                        shuffle_prod = coef + 'x' + shuffle_product(moment_1, moment_2)
                        new_res.append(shuffle_prod.replace('|', '|' + coef + 'x'))
            times += 1
            res = new_res.copy()
        res = '|'.join(res)
        res = res.split('|')
        times = 0
    else:
        res = [y]
        times = 1
    y_split = y.split('|')
    while times <= y_times:
        new_res = []
        for element in res:
            temp = element.split('|')
            for moment_1_repr in temp:
                if 'x' not in moment_1_repr:
                    if moment_1_repr.startswith('-'):
                        moment_1, coef_1 = moment_1_repr[1:], sym.Rational('-1')
                    else:
                        moment_1, coef_1 = moment_1_repr, sym.Rational('1')
                else:
                    split_moment_1 = moment_1_repr.split('x')
                    moment_1, coef_1 = split_moment_1[1], sym.Rational(split_moment_1[0])
                for moment_2_repr in y_split:
                    if 'x' not in moment_2_repr:
                        if moment_2_repr.startswith('-'):
                            moment_2, coef_2 = moment_2_repr[1:], sym.Rational('-1')
                        else:
                            moment_2, coef_2 = moment_2_repr, sym.Rational('1')
                    else:
                        split_moment_2 = moment_2_repr.split('x')
                        moment_2, coef_2 = split_moment_2[1], sym.Rational(split_moment_2[0])
                    coef = str(coef_1 * coef_2)
                    shuffle_prod = coef + 'x' + shuffle_product(moment_1, moment_2)
                    new_res.append(shuffle_prod.replace('|', '|' + coef + 'x'))
        times += 1
        res = new_res.copy()
    return '|'.join(res)


def calculate_combinations(current_order, previous_elements):
    length = len(previous_elements)
    res = []
    for ind in range((current_order + 1) ** length):
        degrees = []
        divisor = ind
        sum_degr = 0
        for i in range(1, length + 1):
            denom = (current_order + 1) ** (length - i)
            value = divisor // denom
            degrees.append(value)
            divisor = divisor % denom
            sum_degr += previous_elements[i - 1]['order'] * value
        if sum_degr == current_order:
            res.append(degrees)
    return res
