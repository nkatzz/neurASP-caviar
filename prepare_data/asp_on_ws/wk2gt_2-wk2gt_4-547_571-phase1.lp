
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

	orientation(wk2gt_2,142,547). coords(wk2gt_2,99,63,547).  visible(wk2gt_2,547). happensAt(inactive(wk2gt_2),547). orientation(wk2gt_4,149,547). coords(wk2gt_4,49,101,547).  visible(wk2gt_4,547). happensAt(walking(wk2gt_4),547).
orientation(wk2gt_2,142,548). coords(wk2gt_2,99,63,548).  visible(wk2gt_2,548). happensAt(inactive(wk2gt_2),548). orientation(wk2gt_4,149,548). coords(wk2gt_4,49,101,548).  visible(wk2gt_4,548). happensAt(walking(wk2gt_4),548).
orientation(wk2gt_2,142,549). coords(wk2gt_2,99,63,549).  visible(wk2gt_2,549). happensAt(inactive(wk2gt_2),549). orientation(wk2gt_4,149,549). coords(wk2gt_4,49,101,549).  visible(wk2gt_4,549). happensAt(walking(wk2gt_4),549).
orientation(wk2gt_2,142,550). coords(wk2gt_2,99,63,550).  visible(wk2gt_2,550). happensAt(inactive(wk2gt_2),550). orientation(wk2gt_4,149,550). coords(wk2gt_4,49,101,550).  visible(wk2gt_4,550). happensAt(walking(wk2gt_4),550).
orientation(wk2gt_2,142,551). coords(wk2gt_2,99,63,551).  visible(wk2gt_2,551). happensAt(inactive(wk2gt_2),551). orientation(wk2gt_4,149,551). coords(wk2gt_4,46,101,551).  visible(wk2gt_4,551). happensAt(walking(wk2gt_4),551).
orientation(wk2gt_2,142,552). coords(wk2gt_2,99,63,552).  visible(wk2gt_2,552). happensAt(inactive(wk2gt_2),552). orientation(wk2gt_4,149,552). coords(wk2gt_4,46,101,552).  visible(wk2gt_4,552). happensAt(walking(wk2gt_4),552).
orientation(wk2gt_2,142,553). coords(wk2gt_2,99,63,553).  visible(wk2gt_2,553). happensAt(inactive(wk2gt_2),553). orientation(wk2gt_4,149,553). coords(wk2gt_4,45,101,553).  visible(wk2gt_4,553). happensAt(walking(wk2gt_4),553).
orientation(wk2gt_2,142,554). coords(wk2gt_2,99,63,554).  visible(wk2gt_2,554). happensAt(inactive(wk2gt_2),554). orientation(wk2gt_4,149,554). coords(wk2gt_4,45,101,554).  visible(wk2gt_4,554). happensAt(walking(wk2gt_4),554).
orientation(wk2gt_2,142,555). coords(wk2gt_2,99,63,555).  visible(wk2gt_2,555). happensAt(inactive(wk2gt_2),555). orientation(wk2gt_4,149,555). coords(wk2gt_4,45,101,555).  visible(wk2gt_4,555). happensAt(walking(wk2gt_4),555).
orientation(wk2gt_2,142,556). coords(wk2gt_2,99,63,556).  visible(wk2gt_2,556). happensAt(inactive(wk2gt_2),556). orientation(wk2gt_4,147,556). coords(wk2gt_4,45,103,556).  visible(wk2gt_4,556). happensAt(walking(wk2gt_4),556).
orientation(wk2gt_2,142,557). coords(wk2gt_2,99,63,557).  visible(wk2gt_2,557). happensAt(inactive(wk2gt_2),557). orientation(wk2gt_4,147,557). coords(wk2gt_4,44,103,557).  visible(wk2gt_4,557). happensAt(walking(wk2gt_4),557).
orientation(wk2gt_2,142,558). coords(wk2gt_2,99,63,558).  visible(wk2gt_2,558). happensAt(inactive(wk2gt_2),558). orientation(wk2gt_4,147,558). coords(wk2gt_4,44,103,558).  visible(wk2gt_4,558). happensAt(walking(wk2gt_4),558).
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
#show holdsAt/2.