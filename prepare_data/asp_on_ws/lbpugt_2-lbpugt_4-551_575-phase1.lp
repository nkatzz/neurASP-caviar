
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

	orientation(lbpugt_2,132,551). coords(lbpugt_2,181,49,551).  visible(lbpugt_2,551). happensAt(active(lbpugt_2),551). orientation(lbpugt_4,135,551). coords(lbpugt_4,219,34,551).  visible(lbpugt_4,551). happensAt(inactive(lbpugt_4),551).
orientation(lbpugt_2,132,552). coords(lbpugt_2,180,49,552).  visible(lbpugt_2,552). happensAt(active(lbpugt_2),552). orientation(lbpugt_4,135,552). coords(lbpugt_4,219,34,552).  visible(lbpugt_4,552). happensAt(inactive(lbpugt_4),552).
orientation(lbpugt_2,132,553). coords(lbpugt_2,180,49,553).  visible(lbpugt_2,553). happensAt(active(lbpugt_2),553). orientation(lbpugt_4,135,553). coords(lbpugt_4,219,34,553).  visible(lbpugt_4,553). happensAt(inactive(lbpugt_4),553).
orientation(lbpugt_2,132,554). coords(lbpugt_2,180,49,554).  visible(lbpugt_2,554). happensAt(active(lbpugt_2),554). orientation(lbpugt_4,135,554). coords(lbpugt_4,219,34,554).  visible(lbpugt_4,554). happensAt(inactive(lbpugt_4),554).
orientation(lbpugt_2,132,555). coords(lbpugt_2,180,49,555).  visible(lbpugt_2,555). happensAt(active(lbpugt_2),555). orientation(lbpugt_4,135,555). coords(lbpugt_4,219,34,555).  visible(lbpugt_4,555). happensAt(inactive(lbpugt_4),555).
orientation(lbpugt_2,132,556). coords(lbpugt_2,180,49,556).  visible(lbpugt_2,556). happensAt(active(lbpugt_2),556). orientation(lbpugt_4,135,556). coords(lbpugt_4,219,34,556).  visible(lbpugt_4,556). happensAt(inactive(lbpugt_4),556).
orientation(lbpugt_2,132,557). coords(lbpugt_2,180,49,557).  visible(lbpugt_2,557). happensAt(active(lbpugt_2),557). orientation(lbpugt_4,135,557). coords(lbpugt_4,219,34,557).  visible(lbpugt_4,557). happensAt(inactive(lbpugt_4),557).
orientation(lbpugt_2,132,558). coords(lbpugt_2,180,49,558).  visible(lbpugt_2,558). happensAt(active(lbpugt_2),558). orientation(lbpugt_4,135,558). coords(lbpugt_4,219,34,558).  visible(lbpugt_4,558). happensAt(inactive(lbpugt_4),558).
orientation(lbpugt_2,132,559). coords(lbpugt_2,180,49,559).  visible(lbpugt_2,559). happensAt(active(lbpugt_2),559). orientation(lbpugt_4,135,559). coords(lbpugt_4,219,34,559).  visible(lbpugt_4,559). happensAt(inactive(lbpugt_4),559).
orientation(lbpugt_2,132,560). coords(lbpugt_2,180,49,560).  visible(lbpugt_2,560). happensAt(active(lbpugt_2),560). orientation(lbpugt_4,135,560). coords(lbpugt_4,219,34,560).  visible(lbpugt_4,560). happensAt(inactive(lbpugt_4),560).
orientation(lbpugt_2,132,561). coords(lbpugt_2,180,49,561).  visible(lbpugt_2,561). happensAt(active(lbpugt_2),561). orientation(lbpugt_4,135,561). coords(lbpugt_4,219,34,561).  visible(lbpugt_4,561). happensAt(inactive(lbpugt_4),561).
orientation(lbpugt_2,132,562). coords(lbpugt_2,180,49,562).  visible(lbpugt_2,562). happensAt(active(lbpugt_2),562). orientation(lbpugt_4,135,562). coords(lbpugt_4,219,34,562).  visible(lbpugt_4,562). happensAt(inactive(lbpugt_4),562).
orientation(lbpugt_2,132,563). coords(lbpugt_2,180,49,563).  visible(lbpugt_2,563). happensAt(active(lbpugt_2),563). orientation(lbpugt_4,135,563). coords(lbpugt_4,219,34,563).  visible(lbpugt_4,563). happensAt(inactive(lbpugt_4),563).
orientation(lbpugt_2,132,564). coords(lbpugt_2,181,49,564).  visible(lbpugt_2,564). happensAt(inactive(lbpugt_2),564). orientation(lbpugt_4,135,564). coords(lbpugt_4,219,34,564).  visible(lbpugt_4,564). happensAt(inactive(lbpugt_4),564).
orientation(lbpugt_2,132,565). coords(lbpugt_2,181,49,565).  visible(lbpugt_2,565). happensAt(inactive(lbpugt_2),565). orientation(lbpugt_4,135,565). coords(lbpugt_4,219,34,565).  visible(lbpugt_4,565). happensAt(inactive(lbpugt_4),565).
orientation(lbpugt_2,132,566). coords(lbpugt_2,181,49,566).  visible(lbpugt_2,566). happensAt(inactive(lbpugt_2),566). orientation(lbpugt_4,135,566). coords(lbpugt_4,219,34,566).  visible(lbpugt_4,566). happensAt(inactive(lbpugt_4),566).
orientation(lbpugt_2,132,567). coords(lbpugt_2,181,49,567).  visible(lbpugt_2,567). happensAt(inactive(lbpugt_2),567). orientation(lbpugt_4,135,567). coords(lbpugt_4,219,34,567).  visible(lbpugt_4,567). happensAt(inactive(lbpugt_4),567).
orientation(lbpugt_2,132,568). coords(lbpugt_2,181,49,568).  visible(lbpugt_2,568). happensAt(inactive(lbpugt_2),568). orientation(lbpugt_4,135,568). coords(lbpugt_4,219,34,568).  visible(lbpugt_4,568). happensAt(inactive(lbpugt_4),568).
orientation(lbpugt_2,132,569). coords(lbpugt_2,181,49,569).  visible(lbpugt_2,569). happensAt(inactive(lbpugt_2),569). orientation(lbpugt_4,135,569). coords(lbpugt_4,219,34,569).  visible(lbpugt_4,569). happensAt(inactive(lbpugt_4),569).
orientation(lbpugt_2,132,570). coords(lbpugt_2,181,49,570).  visible(lbpugt_2,570). happensAt(inactive(lbpugt_2),570). orientation(lbpugt_4,135,570). coords(lbpugt_4,219,34,570).  visible(lbpugt_4,570). happensAt(inactive(lbpugt_4),570).
orientation(lbpugt_2,132,571). coords(lbpugt_2,181,49,571).  visible(lbpugt_2,571). happensAt(inactive(lbpugt_2),571). orientation(lbpugt_4,135,571). coords(lbpugt_4,219,34,571).  visible(lbpugt_4,571). happensAt(inactive(lbpugt_4),571).
orientation(lbpugt_2,132,572). coords(lbpugt_2,181,49,572).  visible(lbpugt_2,572). happensAt(inactive(lbpugt_2),572). orientation(lbpugt_4,135,572). coords(lbpugt_4,219,34,572).  visible(lbpugt_4,572). happensAt(inactive(lbpugt_4),572).
orientation(lbpugt_2,132,573). coords(lbpugt_2,181,49,573).  visible(lbpugt_2,573). happensAt(inactive(lbpugt_2),573). orientation(lbpugt_4,135,573). coords(lbpugt_4,219,34,573).  visible(lbpugt_4,573). happensAt(inactive(lbpugt_4),573).
orientation(lbpugt_2,132,574). coords(lbpugt_2,181,49,574).  visible(lbpugt_2,574). happensAt(inactive(lbpugt_2),574). orientation(lbpugt_4,135,574). coords(lbpugt_4,219,34,574).  visible(lbpugt_4,574). happensAt(inactive(lbpugt_4),574).
orientation(lbpugt_2,132,575). coords(lbpugt_2,181,50,575).  visible(lbpugt_2,575). happensAt(inactive(lbpugt_2),575). orientation(lbpugt_4,135,575). coords(lbpugt_4,219,34,575).  visible(lbpugt_4,575). happensAt(inactive(lbpugt_4),575).
#show holdsAt/2.