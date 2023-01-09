
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

	orientation(lb2gt_4,154,551). coords(lb2gt_4,113,54,551).  visible(lb2gt_4,551). happensAt(walking(lb2gt_4),551). orientation(lb2gt_6,151,551). coords(lb2gt_6,72,122,551).  visible(lb2gt_6,551). happensAt(inactive(lb2gt_6),551).
orientation(lb2gt_4,154,552). coords(lb2gt_4,114,53,552).  visible(lb2gt_4,552). happensAt(walking(lb2gt_4),552). orientation(lb2gt_6,151,552). coords(lb2gt_6,72,122,552).  visible(lb2gt_6,552). happensAt(inactive(lb2gt_6),552).
orientation(lb2gt_4,154,553). coords(lb2gt_4,114,52,553).  visible(lb2gt_4,553). happensAt(walking(lb2gt_4),553). orientation(lb2gt_6,151,553). coords(lb2gt_6,72,122,553).  visible(lb2gt_6,553). happensAt(inactive(lb2gt_6),553).
orientation(lb2gt_4,154,554). coords(lb2gt_4,115,52,554).  visible(lb2gt_4,554). happensAt(walking(lb2gt_4),554). orientation(lb2gt_6,151,554). coords(lb2gt_6,72,122,554).  visible(lb2gt_6,554). happensAt(inactive(lb2gt_6),554).
orientation(lb2gt_4,154,555). coords(lb2gt_4,115,51,555).  visible(lb2gt_4,555). happensAt(walking(lb2gt_4),555). orientation(lb2gt_6,151,555). coords(lb2gt_6,72,122,555).  visible(lb2gt_6,555). happensAt(inactive(lb2gt_6),555).
orientation(lb2gt_4,154,556). coords(lb2gt_4,116,51,556).  visible(lb2gt_4,556). happensAt(walking(lb2gt_4),556). orientation(lb2gt_6,151,556). coords(lb2gt_6,72,122,556).  visible(lb2gt_6,556). happensAt(inactive(lb2gt_6),556).
orientation(lb2gt_4,154,557). coords(lb2gt_4,117,49,557).  visible(lb2gt_4,557). happensAt(walking(lb2gt_4),557). orientation(lb2gt_6,151,557). coords(lb2gt_6,72,122,557).  visible(lb2gt_6,557). happensAt(inactive(lb2gt_6),557).
orientation(lb2gt_4,154,558). coords(lb2gt_4,118,49,558).  visible(lb2gt_4,558). happensAt(walking(lb2gt_4),558). orientation(lb2gt_6,151,558). coords(lb2gt_6,72,122,558).  visible(lb2gt_6,558). happensAt(inactive(lb2gt_6),558).
orientation(lb2gt_4,154,559). coords(lb2gt_4,119,49,559).  visible(lb2gt_4,559). happensAt(walking(lb2gt_4),559). orientation(lb2gt_6,151,559). coords(lb2gt_6,72,122,559).  visible(lb2gt_6,559). happensAt(inactive(lb2gt_6),559).
orientation(lb2gt_4,154,560). coords(lb2gt_4,120,48,560).  visible(lb2gt_4,560). happensAt(walking(lb2gt_4),560). orientation(lb2gt_6,151,560). coords(lb2gt_6,72,122,560).  visible(lb2gt_6,560). happensAt(inactive(lb2gt_6),560).
orientation(lb2gt_4,154,561). coords(lb2gt_4,122,47,561).  visible(lb2gt_4,561). happensAt(walking(lb2gt_4),561). orientation(lb2gt_6,151,561). coords(lb2gt_6,72,122,561).  visible(lb2gt_6,561). happensAt(inactive(lb2gt_6),561).
orientation(lb2gt_4,154,562). coords(lb2gt_4,124,47,562).  visible(lb2gt_4,562). happensAt(walking(lb2gt_4),562). orientation(lb2gt_6,151,562). coords(lb2gt_6,72,122,562).  visible(lb2gt_6,562). happensAt(inactive(lb2gt_6),562).
orientation(lb2gt_4,154,563). coords(lb2gt_4,124,47,563).  visible(lb2gt_4,563). happensAt(walking(lb2gt_4),563). orientation(lb2gt_6,151,563). coords(lb2gt_6,72,122,563).  visible(lb2gt_6,563). happensAt(inactive(lb2gt_6),563).
orientation(lb2gt_4,154,564). coords(lb2gt_4,125,47,564).  visible(lb2gt_4,564). happensAt(walking(lb2gt_4),564). orientation(lb2gt_6,151,564). coords(lb2gt_6,72,122,564).  visible(lb2gt_6,564). happensAt(inactive(lb2gt_6),564).
orientation(lb2gt_4,154,565). coords(lb2gt_4,125,46,565).  visible(lb2gt_4,565). happensAt(walking(lb2gt_4),565). orientation(lb2gt_6,151,565). coords(lb2gt_6,72,122,565).  visible(lb2gt_6,565). happensAt(inactive(lb2gt_6),565).
orientation(lb2gt_4,154,566). coords(lb2gt_4,126,46,566).  visible(lb2gt_4,566). happensAt(walking(lb2gt_4),566). orientation(lb2gt_6,151,566). coords(lb2gt_6,72,122,566).  visible(lb2gt_6,566). happensAt(inactive(lb2gt_6),566).
orientation(lb2gt_4,154,567). coords(lb2gt_4,126,45,567).  visible(lb2gt_4,567). happensAt(walking(lb2gt_4),567). orientation(lb2gt_6,151,567). coords(lb2gt_6,72,122,567).  visible(lb2gt_6,567). happensAt(inactive(lb2gt_6),567).
orientation(lb2gt_4,154,568). coords(lb2gt_4,126,45,568).  visible(lb2gt_4,568). happensAt(walking(lb2gt_4),568). orientation(lb2gt_6,151,568). coords(lb2gt_6,72,122,568).  visible(lb2gt_6,568). happensAt(inactive(lb2gt_6),568).
orientation(lb2gt_4,154,569). coords(lb2gt_4,127,44,569).  visible(lb2gt_4,569). happensAt(walking(lb2gt_4),569). orientation(lb2gt_6,151,569). coords(lb2gt_6,72,122,569).  visible(lb2gt_6,569). happensAt(inactive(lb2gt_6),569).
orientation(lb2gt_4,154,570). coords(lb2gt_4,128,44,570).  visible(lb2gt_4,570). happensAt(walking(lb2gt_4),570). orientation(lb2gt_6,151,570). coords(lb2gt_6,72,122,570).  visible(lb2gt_6,570). happensAt(inactive(lb2gt_6),570).
orientation(lb2gt_4,154,571). coords(lb2gt_4,130,43,571).  visible(lb2gt_4,571). happensAt(walking(lb2gt_4),571). orientation(lb2gt_6,151,571). coords(lb2gt_6,72,122,571).  visible(lb2gt_6,571). happensAt(inactive(lb2gt_6),571).
orientation(lb2gt_4,154,572). coords(lb2gt_4,131,43,572).  visible(lb2gt_4,572). happensAt(walking(lb2gt_4),572). orientation(lb2gt_6,151,572). coords(lb2gt_6,72,122,572).  visible(lb2gt_6,572). happensAt(inactive(lb2gt_6),572).
orientation(lb2gt_4,154,573). coords(lb2gt_4,133,43,573).  visible(lb2gt_4,573). happensAt(walking(lb2gt_4),573). orientation(lb2gt_6,151,573). coords(lb2gt_6,72,122,573).  visible(lb2gt_6,573). happensAt(inactive(lb2gt_6),573).
orientation(lb2gt_4,154,574). coords(lb2gt_4,133,43,574).  visible(lb2gt_4,574). happensAt(walking(lb2gt_4),574). orientation(lb2gt_6,151,574). coords(lb2gt_6,72,122,574).  visible(lb2gt_6,574). happensAt(inactive(lb2gt_6),574).
orientation(lb2gt_4,154,575). coords(lb2gt_4,133,42,575).  visible(lb2gt_4,575). happensAt(walking(lb2gt_4),575). orientation(lb2gt_6,151,575). coords(lb2gt_6,72,122,575).  visible(lb2gt_6,575). happensAt(inactive(lb2gt_6),575).
#show holdsAt/2.