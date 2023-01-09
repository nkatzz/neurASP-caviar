

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,636). coords(wk2gt_1,115,53,636).  visible(wk2gt_1,636). orientation(wk2gt_2,142,636). coords(wk2gt_2,99,63,636).  visible(wk2gt_2,636).
orientation(wk2gt_1,125,637). coords(wk2gt_1,115,53,637).  visible(wk2gt_1,637). orientation(wk2gt_2,142,637). coords(wk2gt_2,99,63,637).  visible(wk2gt_2,637).
orientation(wk2gt_1,125,638). coords(wk2gt_1,115,53,638).  visible(wk2gt_1,638). orientation(wk2gt_2,142,638). coords(wk2gt_2,99,63,638).  visible(wk2gt_2,638).
orientation(wk2gt_1,125,639). coords(wk2gt_1,115,53,639).  visible(wk2gt_1,639). orientation(wk2gt_2,142,639). coords(wk2gt_2,99,63,639).  visible(wk2gt_2,639).
orientation(wk2gt_1,125,640). coords(wk2gt_1,115,53,640).  visible(wk2gt_1,640). orientation(wk2gt_2,142,640). coords(wk2gt_2,99,63,640).  visible(wk2gt_2,640).
orientation(wk2gt_1,125,641). coords(wk2gt_1,115,53,641).  visible(wk2gt_1,641). orientation(wk2gt_2,142,641). coords(wk2gt_2,99,63,641).  visible(wk2gt_2,641).
orientation(wk2gt_1,125,642). coords(wk2gt_1,114,53,642).  visible(wk2gt_1,642). orientation(wk2gt_2,142,642). coords(wk2gt_2,99,63,642).  visible(wk2gt_2,642).
orientation(wk2gt_1,125,643). coords(wk2gt_1,114,53,643).  visible(wk2gt_1,643). orientation(wk2gt_2,142,643). coords(wk2gt_2,99,63,643).  visible(wk2gt_2,643).
orientation(wk2gt_1,125,644). coords(wk2gt_1,114,53,644).  visible(wk2gt_1,644). orientation(wk2gt_2,142,644). coords(wk2gt_2,99,63,644).  visible(wk2gt_2,644).
orientation(wk2gt_1,125,645). coords(wk2gt_1,114,53,645).  visible(wk2gt_1,645). orientation(wk2gt_2,142,645). coords(wk2gt_2,99,63,645).  visible(wk2gt_2,645).
orientation(wk2gt_1,125,646). coords(wk2gt_1,114,53,646).  visible(wk2gt_1,646). orientation(wk2gt_2,142,646). coords(wk2gt_2,99,63,646).  visible(wk2gt_2,646).
orientation(wk2gt_1,125,647). coords(wk2gt_1,114,53,647).  visible(wk2gt_1,647). orientation(wk2gt_2,142,647). coords(wk2gt_2,99,63,647).  visible(wk2gt_2,647).
orientation(wk2gt_1,125,648). coords(wk2gt_1,114,53,648).  visible(wk2gt_1,648). orientation(wk2gt_2,142,648). coords(wk2gt_2,99,63,648).  visible(wk2gt_2,648).
orientation(wk2gt_1,125,649). coords(wk2gt_1,114,53,649).  visible(wk2gt_1,649). orientation(wk2gt_2,142,649). coords(wk2gt_2,99,63,649).  visible(wk2gt_2,649).
orientation(wk2gt_1,125,650). coords(wk2gt_1,114,53,650).  visible(wk2gt_1,650). orientation(wk2gt_2,142,650). coords(wk2gt_2,99,63,650).  visible(wk2gt_2,650).
orientation(wk2gt_1,125,651). coords(wk2gt_1,114,53,651).  visible(wk2gt_1,651). orientation(wk2gt_2,142,651). coords(wk2gt_2,99,63,651).  visible(wk2gt_2,651).
orientation(wk2gt_1,125,652). coords(wk2gt_1,114,53,652).  visible(wk2gt_1,652). orientation(wk2gt_2,142,652). coords(wk2gt_2,99,63,652).  visible(wk2gt_2,652).
orientation(wk2gt_1,125,653). coords(wk2gt_1,114,53,653).  visible(wk2gt_1,653). orientation(wk2gt_2,142,653). coords(wk2gt_2,99,63,653).  visible(wk2gt_2,653).
orientation(wk2gt_1,125,654). coords(wk2gt_1,114,53,654).  visible(wk2gt_1,654). orientation(wk2gt_2,142,654). coords(wk2gt_2,99,63,654).  visible(wk2gt_2,654).
orientation(wk2gt_1,125,655). coords(wk2gt_1,114,53,655).  visible(wk2gt_1,655). orientation(wk2gt_2,142,655). coords(wk2gt_2,99,63,655).  visible(wk2gt_2,655).
orientation(wk2gt_1,125,656). coords(wk2gt_1,114,53,656).  visible(wk2gt_1,656). orientation(wk2gt_2,142,656). coords(wk2gt_2,99,63,656).  visible(wk2gt_2,656).
orientation(wk2gt_1,125,657). coords(wk2gt_1,114,53,657).  visible(wk2gt_1,657). orientation(wk2gt_2,142,657). coords(wk2gt_2,99,63,657).  visible(wk2gt_2,657).
orientation(wk2gt_1,125,658). coords(wk2gt_1,114,53,658).  visible(wk2gt_1,658). orientation(wk2gt_2,142,658). coords(wk2gt_2,99,63,658).  visible(wk2gt_2,658).
orientation(wk2gt_1,125,659). coords(wk2gt_1,114,53,659).  visible(wk2gt_1,659). orientation(wk2gt_2,142,659). coords(wk2gt_2,99,63,659).  visible(wk2gt_2,659).
orientation(wk2gt_1,125,660). coords(wk2gt_1,114,53,660).  visible(wk2gt_1,660). orientation(wk2gt_2,142,660). coords(wk2gt_2,99,63,660).  visible(wk2gt_2,660).
orientation(wk2gt_1,125,661). coords(wk2gt_1,114,53,661).  visible(wk2gt_1,661). orientation(wk2gt_2,142,661). coords(wk2gt_2,99,63,661).  visible(wk2gt_2,661).
happensAt(disappear(wk2gt_1),660). happensAt(disappear(wk2gt_2),660).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=636..660.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=636..660.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),636). :- holdsAt(meeting(wk2gt_2,wk2gt_1),636).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),637). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),637).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),638). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),638).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),639). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),639).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),640). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),640).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),641). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),641).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),642). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),642).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),643). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),643).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),644). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),644).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),645). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),645).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),646). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),646).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),647). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),647).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),648). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),648).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),649). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),649).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),650). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),650).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),651). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),651).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),652). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),652).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),653). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),653).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),654). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),654).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),655). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),655).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),656). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),656).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),657). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),657).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),658). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),658).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),659). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),659).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),660). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),660).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),661). :- holdsAt(meeting(wk2gt_2,wk2gt_1),661).
:- simpleEvent(T,P,walking), T>636 , T<660.
:- simpleEvent(T,P,running), T>636 , T<660.

#show simpleEvent/3.