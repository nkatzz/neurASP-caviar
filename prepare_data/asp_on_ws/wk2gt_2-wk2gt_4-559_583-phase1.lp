
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(wk2gt_2,142,559). coords(wk2gt_2,99,63,559).  visible(wk2gt_2,559). happensAt(inactive(wk2gt_2),559). orientation(wk2gt_4,147,559). coords(wk2gt_4,44,104,559).  visible(wk2gt_4,559). happensAt(walking(wk2gt_4),559).
orientation(wk2gt_2,142,560). coords(wk2gt_2,99,63,560).  visible(wk2gt_2,560). happensAt(inactive(wk2gt_2),560). orientation(wk2gt_4,147,560). coords(wk2gt_4,44,104,560).  visible(wk2gt_4,560). happensAt(walking(wk2gt_4),560).
orientation(wk2gt_2,142,561). coords(wk2gt_2,99,63,561).  visible(wk2gt_2,561). happensAt(inactive(wk2gt_2),561). orientation(wk2gt_4,147,561). coords(wk2gt_4,44,104,561).  visible(wk2gt_4,561). happensAt(walking(wk2gt_4),561).
orientation(wk2gt_2,142,562). coords(wk2gt_2,99,63,562).  visible(wk2gt_2,562). happensAt(inactive(wk2gt_2),562). orientation(wk2gt_4,147,562). coords(wk2gt_4,44,104,562).  visible(wk2gt_4,562). happensAt(walking(wk2gt_4),562).
orientation(wk2gt_2,142,563). coords(wk2gt_2,99,63,563).  visible(wk2gt_2,563). happensAt(inactive(wk2gt_2),563). orientation(wk2gt_4,147,563). coords(wk2gt_4,43,104,563).  visible(wk2gt_4,563). happensAt(walking(wk2gt_4),563).
orientation(wk2gt_2,142,564). coords(wk2gt_2,99,63,564).  visible(wk2gt_2,564). happensAt(inactive(wk2gt_2),564). orientation(wk2gt_4,147,564). coords(wk2gt_4,43,104,564).  visible(wk2gt_4,564). happensAt(walking(wk2gt_4),564).
orientation(wk2gt_2,142,565). coords(wk2gt_2,99,63,565).  visible(wk2gt_2,565). happensAt(inactive(wk2gt_2),565). orientation(wk2gt_4,147,565). coords(wk2gt_4,43,104,565).  visible(wk2gt_4,565). happensAt(walking(wk2gt_4),565).
orientation(wk2gt_2,142,566). coords(wk2gt_2,99,63,566).  visible(wk2gt_2,566). happensAt(inactive(wk2gt_2),566). orientation(wk2gt_4,147,566). coords(wk2gt_4,43,104,566).  visible(wk2gt_4,566). happensAt(walking(wk2gt_4),566).
orientation(wk2gt_2,142,567). coords(wk2gt_2,99,63,567).  visible(wk2gt_2,567). happensAt(inactive(wk2gt_2),567). orientation(wk2gt_4,147,567). coords(wk2gt_4,43,105,567).  visible(wk2gt_4,567). happensAt(walking(wk2gt_4),567).
orientation(wk2gt_2,142,568). coords(wk2gt_2,99,63,568).  visible(wk2gt_2,568). happensAt(inactive(wk2gt_2),568). orientation(wk2gt_4,147,568). coords(wk2gt_4,42,105,568).  visible(wk2gt_4,568). happensAt(walking(wk2gt_4),568).
orientation(wk2gt_2,142,569). coords(wk2gt_2,99,63,569).  visible(wk2gt_2,569). happensAt(inactive(wk2gt_2),569). orientation(wk2gt_4,147,569). coords(wk2gt_4,42,105,569).  visible(wk2gt_4,569). happensAt(walking(wk2gt_4),569).
orientation(wk2gt_2,142,570). coords(wk2gt_2,99,63,570).  visible(wk2gt_2,570). happensAt(inactive(wk2gt_2),570). orientation(wk2gt_4,147,570). coords(wk2gt_4,42,105,570).  visible(wk2gt_4,570). happensAt(walking(wk2gt_4),570).
orientation(wk2gt_2,142,571). coords(wk2gt_2,99,63,571).  visible(wk2gt_2,571). happensAt(inactive(wk2gt_2),571). orientation(wk2gt_4,147,571). coords(wk2gt_4,42,105,571).  visible(wk2gt_4,571). happensAt(walking(wk2gt_4),571).
orientation(wk2gt_2,142,572). coords(wk2gt_2,99,63,572).  visible(wk2gt_2,572). happensAt(inactive(wk2gt_2),572). orientation(wk2gt_4,147,572). coords(wk2gt_4,42,105,572).  visible(wk2gt_4,572). happensAt(walking(wk2gt_4),572).
orientation(wk2gt_2,142,573). coords(wk2gt_2,99,63,573).  visible(wk2gt_2,573). happensAt(inactive(wk2gt_2),573). orientation(wk2gt_4,147,573). coords(wk2gt_4,42,105,573).  visible(wk2gt_4,573). happensAt(walking(wk2gt_4),573).
orientation(wk2gt_2,142,574). coords(wk2gt_2,99,63,574).  visible(wk2gt_2,574). happensAt(inactive(wk2gt_2),574). orientation(wk2gt_4,147,574). coords(wk2gt_4,42,105,574).  visible(wk2gt_4,574). happensAt(walking(wk2gt_4),574).
orientation(wk2gt_2,142,575). coords(wk2gt_2,99,63,575).  visible(wk2gt_2,575). happensAt(inactive(wk2gt_2),575). orientation(wk2gt_4,147,575). coords(wk2gt_4,42,107,575).  visible(wk2gt_4,575). happensAt(walking(wk2gt_4),575).
orientation(wk2gt_2,142,576). coords(wk2gt_2,99,63,576).  visible(wk2gt_2,576). happensAt(inactive(wk2gt_2),576). orientation(wk2gt_4,152,576). coords(wk2gt_4,42,107,576).  visible(wk2gt_4,576). happensAt(walking(wk2gt_4),576).
orientation(wk2gt_2,142,577). coords(wk2gt_2,99,63,577).  visible(wk2gt_2,577). happensAt(inactive(wk2gt_2),577). orientation(wk2gt_4,152,577). coords(wk2gt_4,42,107,577).  visible(wk2gt_4,577). happensAt(walking(wk2gt_4),577).
orientation(wk2gt_2,142,578). coords(wk2gt_2,99,63,578).  visible(wk2gt_2,578). happensAt(inactive(wk2gt_2),578). orientation(wk2gt_4,152,578). coords(wk2gt_4,40,107,578).  visible(wk2gt_4,578). happensAt(walking(wk2gt_4),578).
orientation(wk2gt_2,142,579). coords(wk2gt_2,99,63,579).  visible(wk2gt_2,579). happensAt(inactive(wk2gt_2),579). orientation(wk2gt_4,152,579). coords(wk2gt_4,40,107,579).  visible(wk2gt_4,579). happensAt(walking(wk2gt_4),579).
orientation(wk2gt_2,142,580). coords(wk2gt_2,99,63,580).  visible(wk2gt_2,580). happensAt(inactive(wk2gt_2),580). orientation(wk2gt_4,152,580). coords(wk2gt_4,40,107,580).  visible(wk2gt_4,580). happensAt(walking(wk2gt_4),580).
orientation(wk2gt_2,142,581). coords(wk2gt_2,99,63,581).  visible(wk2gt_2,581). happensAt(inactive(wk2gt_2),581). orientation(wk2gt_4,152,581). coords(wk2gt_4,39,107,581).  visible(wk2gt_4,581). happensAt(walking(wk2gt_4),581).
orientation(wk2gt_2,142,582). coords(wk2gt_2,99,63,582).  visible(wk2gt_2,582). happensAt(inactive(wk2gt_2),582). orientation(wk2gt_4,152,582). coords(wk2gt_4,39,107,582).  visible(wk2gt_4,582). happensAt(walking(wk2gt_4),582).
orientation(wk2gt_2,142,583). coords(wk2gt_2,99,63,583).  visible(wk2gt_2,583). happensAt(inactive(wk2gt_2),583). orientation(wk2gt_4,152,583). coords(wk2gt_4,39,108,583).  visible(wk2gt_4,583). happensAt(walking(wk2gt_4),583).
#show holdsAt/2.