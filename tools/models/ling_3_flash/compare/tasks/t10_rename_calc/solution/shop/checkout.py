from shop import calc as pricing


def receipt(items, tax_rate=0.0):
    subtotal = pricing.compute_total(items)
    tax = round(subtotal * tax_rate, 2)
    return {"subtotal": subtotal, "tax": tax, "total": round(subtotal + tax, 2)}
