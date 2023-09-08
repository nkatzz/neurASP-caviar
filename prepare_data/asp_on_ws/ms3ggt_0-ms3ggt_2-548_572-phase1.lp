
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

	orientation(ms3ggt_0,168,548). coords(ms3ggt_0,73,191,548).  visible(ms3ggt_0,548). happensAt(inactive(ms3ggt_0),548). orientation(ms3ggt_2,142,548). coords(ms3ggt_2,84,96,548).  visible(ms3ggt_2,548). happensAt(walking(ms3ggt_2),548).
orientation(ms3ggt_0,168,549). coords(ms3ggt_0,73,191,549).  visible(ms3ggt_0,549). happensAt(inactive(ms3ggt_0),549). orientation(ms3ggt_2,142,549). coords(ms3ggt_2,84,96,549).  visible(ms3ggt_2,549). happensAt(walking(ms3ggt_2),549).
orientation(ms3ggt_0,168,550). coords(ms3ggt_0,73,191,550).  visible(ms3ggt_0,550). happensAt(inactive(ms3ggt_0),550). orientation(ms3ggt_2,142,550). coords(ms3ggt_2,84,96,550).  visible(ms3ggt_2,550). happensAt(walking(ms3ggt_2),550).
orientation(ms3ggt_0,168,551). coords(ms3ggt_0,73,191,551).  visible(ms3ggt_0,551). happensAt(inactive(ms3ggt_0),551). orientation(ms3ggt_2,142,551). coords(ms3ggt_2,84,96,551).  visible(ms3ggt_2,551). happensAt(walking(ms3ggt_2),551).
orientation(ms3ggt_0,168,552). coords(ms3ggt_0,73,191,552).  visible(ms3ggt_0,552). happensAt(inactive(ms3ggt_0),552). orientation(ms3ggt_2,142,552). coords(ms3ggt_2,84,95,552).  visible(ms3ggt_2,552). happensAt(walking(ms3ggt_2),552).
orientation(ms3ggt_0,168,553). coords(ms3ggt_0,73,191,553).  visible(ms3ggt_0,553). happensAt(inactive(ms3ggt_0),553). orientation(ms3ggt_2,142,553). coords(ms3ggt_2,84,95,553).  visible(ms3ggt_2,553). happensAt(walking(ms3ggt_2),553).
orientation(ms3ggt_0,168,554). coords(ms3ggt_0,73,191,554).  visible(ms3ggt_0,554). happensAt(inactive(ms3ggt_0),554). orientation(ms3ggt_2,142,554). coords(ms3ggt_2,84,95,554).  visible(ms3ggt_2,554). happensAt(walking(ms3ggt_2),554).
orientation(ms3ggt_0,168,555). coords(ms3ggt_0,73,191,555).  visible(ms3ggt_0,555). happensAt(inactive(ms3ggt_0),555). orientation(ms3ggt_2,142,555). coords(ms3ggt_2,84,95,555).  visible(ms3ggt_2,555). happensAt(walking(ms3ggt_2),555).
orientation(ms3ggt_0,168,556). coords(ms3ggt_0,73,191,556).  visible(ms3ggt_0,556). happensAt(inactive(ms3ggt_0),556). orientation(ms3ggt_2,142,556). coords(ms3ggt_2,84,94,556).  visible(ms3ggt_2,556). happensAt(walking(ms3ggt_2),556).
orientation(ms3ggt_0,168,557). coords(ms3ggt_0,73,191,557).  visible(ms3ggt_0,557). happensAt(inactive(ms3ggt_0),557). orientation(ms3ggt_2,142,557). coords(ms3ggt_2,84,94,557).  visible(ms3ggt_2,557). happensAt(walking(ms3ggt_2),557).
orientation(ms3ggt_0,168,558). coords(ms3ggt_0,73,191,558).  visible(ms3ggt_0,558). happensAt(inactive(ms3ggt_0),558). orientation(ms3ggt_2,142,558). coords(ms3ggt_2,84,93,558).  visible(ms3ggt_2,558). happensAt(walking(ms3ggt_2),558).
orientation(ms3ggt_0,168,559). coords(ms3ggt_0,73,191,559).  visible(ms3ggt_0,559). happensAt(inactive(ms3ggt_0),559). orientation(ms3ggt_2,142,559). coords(ms3ggt_2,84,93,559).  visible(ms3ggt_2,559). happensAt(walking(ms3ggt_2),559).
orientation(ms3ggt_0,168,560). coords(ms3ggt_0,73,191,560).  visible(ms3ggt_0,560). happensAt(inactive(ms3ggt_0),560). orientation(ms3ggt_2,150,560). coords(ms3ggt_2,84,92,560).  visible(ms3ggt_2,560). happensAt(walking(ms3ggt_2),560).
orientation(ms3ggt_0,168,561). coords(ms3ggt_0,73,191,561).  visible(ms3ggt_0,561). happensAt(inactive(ms3ggt_0),561). orientation(ms3ggt_2,150,561). coords(ms3ggt_2,83,91,561).  visible(ms3ggt_2,561). happensAt(walking(ms3ggt_2),561).
orientation(ms3ggt_0,168,562). coords(ms3ggt_0,73,191,562).  visible(ms3ggt_0,562). happensAt(inactive(ms3ggt_0),562). orientation(ms3ggt_2,150,562). coords(ms3ggt_2,83,90,562).  visible(ms3ggt_2,562). happensAt(walking(ms3ggt_2),562).
orientation(ms3ggt_0,168,563). coords(ms3ggt_0,73,191,563).  visible(ms3ggt_0,563). happensAt(inactive(ms3ggt_0),563). orientation(ms3ggt_2,150,563). coords(ms3ggt_2,83,90,563).  visible(ms3ggt_2,563). happensAt(walking(ms3ggt_2),563).
orientation(ms3ggt_0,168,564). coords(ms3ggt_0,73,191,564).  visible(ms3ggt_0,564). happensAt(inactive(ms3ggt_0),564). orientation(ms3ggt_2,150,564). coords(ms3ggt_2,83,90,564).  visible(ms3ggt_2,564). happensAt(walking(ms3ggt_2),564).
orientation(ms3ggt_0,168,565). coords(ms3ggt_0,73,191,565).  visible(ms3ggt_0,565). happensAt(inactive(ms3ggt_0),565). orientation(ms3ggt_2,150,565). coords(ms3ggt_2,83,90,565).  visible(ms3ggt_2,565). happensAt(walking(ms3ggt_2),565).
orientation(ms3ggt_0,168,566). coords(ms3ggt_0,73,191,566).  visible(ms3ggt_0,566). happensAt(inactive(ms3ggt_0),566). orientation(ms3ggt_2,150,566). coords(ms3ggt_2,83,90,566).  visible(ms3ggt_2,566). happensAt(walking(ms3ggt_2),566).
orientation(ms3ggt_0,168,567). coords(ms3ggt_0,73,191,567).  visible(ms3ggt_0,567). happensAt(inactive(ms3ggt_0),567). orientation(ms3ggt_2,150,567). coords(ms3ggt_2,83,90,567).  visible(ms3ggt_2,567). happensAt(walking(ms3ggt_2),567).
orientation(ms3ggt_0,168,568). coords(ms3ggt_0,73,191,568).  visible(ms3ggt_0,568). happensAt(inactive(ms3ggt_0),568). orientation(ms3ggt_2,150,568). coords(ms3ggt_2,83,90,568).  visible(ms3ggt_2,568). happensAt(walking(ms3ggt_2),568).
orientation(ms3ggt_0,168,569). coords(ms3ggt_0,73,192,569).  visible(ms3ggt_0,569). happensAt(inactive(ms3ggt_0),569). orientation(ms3ggt_2,150,569). coords(ms3ggt_2,83,89,569).  visible(ms3ggt_2,569). happensAt(walking(ms3ggt_2),569).
orientation(ms3ggt_0,168,570). coords(ms3ggt_0,73,192,570).  visible(ms3ggt_0,570). happensAt(inactive(ms3ggt_0),570). orientation(ms3ggt_2,150,570). coords(ms3ggt_2,83,89,570).  visible(ms3ggt_2,570). happensAt(walking(ms3ggt_2),570).
orientation(ms3ggt_0,168,571). coords(ms3ggt_0,73,192,571).  visible(ms3ggt_0,571). happensAt(inactive(ms3ggt_0),571). orientation(ms3ggt_2,150,571). coords(ms3ggt_2,83,89,571).  visible(ms3ggt_2,571). happensAt(walking(ms3ggt_2),571).
orientation(ms3ggt_0,168,572). coords(ms3ggt_0,73,192,572).  visible(ms3ggt_0,572). happensAt(inactive(ms3ggt_0),572). orientation(ms3ggt_2,150,572). coords(ms3ggt_2,83,89,572).  visible(ms3ggt_2,572). happensAt(walking(ms3ggt_2),572).
#show holdsAt/2.