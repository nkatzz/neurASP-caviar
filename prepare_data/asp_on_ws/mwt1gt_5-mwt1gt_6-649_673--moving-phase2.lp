

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

	person(mwt1gt_5;mwt1gt_6).
orientation(mwt1gt_5,150,649). coords(mwt1gt_5,82,75,649).  visible(mwt1gt_5,649). orientation(mwt1gt_6,150,649). coords(mwt1gt_6,74,82,649).  visible(mwt1gt_6,649).
orientation(mwt1gt_5,150,650). coords(mwt1gt_5,82,75,650).  visible(mwt1gt_5,650). orientation(mwt1gt_6,150,650). coords(mwt1gt_6,74,82,650).  visible(mwt1gt_6,650).
orientation(mwt1gt_5,150,651). coords(mwt1gt_5,82,75,651).  visible(mwt1gt_5,651). orientation(mwt1gt_6,150,651). coords(mwt1gt_6,74,82,651).  visible(mwt1gt_6,651).
orientation(mwt1gt_5,150,652). coords(mwt1gt_5,82,75,652).  visible(mwt1gt_5,652). orientation(mwt1gt_6,150,652). coords(mwt1gt_6,74,82,652).  visible(mwt1gt_6,652).
orientation(mwt1gt_5,150,653). coords(mwt1gt_5,82,75,653).  visible(mwt1gt_5,653). orientation(mwt1gt_6,150,653). coords(mwt1gt_6,74,82,653).  visible(mwt1gt_6,653).
orientation(mwt1gt_5,150,654). coords(mwt1gt_5,83,78,654).  visible(mwt1gt_5,654). orientation(mwt1gt_6,150,654). coords(mwt1gt_6,74,82,654).  visible(mwt1gt_6,654).
orientation(mwt1gt_5,150,655). coords(mwt1gt_5,83,78,655).  visible(mwt1gt_5,655). orientation(mwt1gt_6,150,655). coords(mwt1gt_6,74,82,655).  visible(mwt1gt_6,655).
orientation(mwt1gt_5,150,656). coords(mwt1gt_5,83,78,656).  visible(mwt1gt_5,656). orientation(mwt1gt_6,150,656). coords(mwt1gt_6,74,82,656).  visible(mwt1gt_6,656).
orientation(mwt1gt_5,150,657). coords(mwt1gt_5,83,78,657).  visible(mwt1gt_5,657). orientation(mwt1gt_6,150,657). coords(mwt1gt_6,75,83,657).  visible(mwt1gt_6,657).
orientation(mwt1gt_5,150,658). coords(mwt1gt_5,83,78,658).  visible(mwt1gt_5,658). orientation(mwt1gt_6,150,658). coords(mwt1gt_6,75,83,658).  visible(mwt1gt_6,658).
orientation(mwt1gt_5,150,659). coords(mwt1gt_5,84,80,659).  visible(mwt1gt_5,659). orientation(mwt1gt_6,150,659). coords(mwt1gt_6,75,84,659).  visible(mwt1gt_6,659).
orientation(mwt1gt_5,150,660). coords(mwt1gt_5,84,80,660).  visible(mwt1gt_5,660). orientation(mwt1gt_6,150,660). coords(mwt1gt_6,75,84,660).  visible(mwt1gt_6,660).
orientation(mwt1gt_5,150,661). coords(mwt1gt_5,84,80,661).  visible(mwt1gt_5,661). orientation(mwt1gt_6,150,661). coords(mwt1gt_6,75,84,661).  visible(mwt1gt_6,661).
orientation(mwt1gt_5,150,662). coords(mwt1gt_5,84,80,662).  visible(mwt1gt_5,662). orientation(mwt1gt_6,150,662). coords(mwt1gt_6,75,84,662).  visible(mwt1gt_6,662).
orientation(mwt1gt_5,150,663). coords(mwt1gt_5,84,80,663).  visible(mwt1gt_5,663). orientation(mwt1gt_6,150,663). coords(mwt1gt_6,75,84,663).  visible(mwt1gt_6,663).
orientation(mwt1gt_5,150,664). coords(mwt1gt_5,84,80,664).  visible(mwt1gt_5,664). orientation(mwt1gt_6,150,664). coords(mwt1gt_6,75,84,664).  visible(mwt1gt_6,664).
orientation(mwt1gt_5,150,665). coords(mwt1gt_5,84,80,665).  visible(mwt1gt_5,665). orientation(mwt1gt_6,150,665). coords(mwt1gt_6,75,84,665).  visible(mwt1gt_6,665).
orientation(mwt1gt_5,150,666). coords(mwt1gt_5,84,80,666).  visible(mwt1gt_5,666). orientation(mwt1gt_6,150,666). coords(mwt1gt_6,75,84,666).  visible(mwt1gt_6,666).
orientation(mwt1gt_5,150,667). coords(mwt1gt_5,84,82,667).  visible(mwt1gt_5,667). orientation(mwt1gt_6,150,667). coords(mwt1gt_6,75,86,667).  visible(mwt1gt_6,667).
orientation(mwt1gt_5,150,668). coords(mwt1gt_5,84,82,668).  visible(mwt1gt_5,668). orientation(mwt1gt_6,150,668). coords(mwt1gt_6,75,86,668).  visible(mwt1gt_6,668).
orientation(mwt1gt_5,150,669). coords(mwt1gt_5,85,83,669).  visible(mwt1gt_5,669). orientation(mwt1gt_6,150,669). coords(mwt1gt_6,75,89,669).  visible(mwt1gt_6,669).
orientation(mwt1gt_5,150,670). coords(mwt1gt_5,85,83,670).  visible(mwt1gt_5,670). orientation(mwt1gt_6,150,670). coords(mwt1gt_6,75,89,670).  visible(mwt1gt_6,670).
orientation(mwt1gt_5,150,671). coords(mwt1gt_5,85,83,671).  visible(mwt1gt_5,671). orientation(mwt1gt_6,150,671). coords(mwt1gt_6,76,90,671).  visible(mwt1gt_6,671).
orientation(mwt1gt_5,150,672). coords(mwt1gt_5,85,83,672).  visible(mwt1gt_5,672). orientation(mwt1gt_6,150,672). coords(mwt1gt_6,76,90,672).  visible(mwt1gt_6,672).
orientation(mwt1gt_5,150,673). coords(mwt1gt_5,85,83,673).  visible(mwt1gt_5,673). orientation(mwt1gt_6,150,673). coords(mwt1gt_6,76,90,673).  visible(mwt1gt_6,673).
orientation(mwt1gt_5,150,674). coords(mwt1gt_5,85,83,674).  visible(mwt1gt_5,674). orientation(mwt1gt_6,150,674). coords(mwt1gt_6,76,90,674).  visible(mwt1gt_6,674).
happensAt(disappear(mwt1gt_5),673). happensAt(disappear(mwt1gt_6),673).
1{simpleEvent(T,mwt1gt_5,active); simpleEvent(T,mwt1gt_5,inactive); simpleEvent(T,mwt1gt_5,walking); simpleEvent(T,mwt1gt_5,running)}1 :- T=649..673.
1{simpleEvent(T,mwt1gt_6,active); simpleEvent(T,mwt1gt_6,inactive); simpleEvent(T,mwt1gt_6,walking); simpleEvent(T,mwt1gt_6,running)}1 :- T=649..673.
:- holdsAt(moving(mwt1gt_5,mwt1gt_6),649). :- holdsAt(moving(mwt1gt_6,mwt1gt_5),649).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),650). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),650).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),651). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),651).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),652). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),652).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),653). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),653).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),654). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),654).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),655). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),655).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),656). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),656).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),657). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),657).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),658). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),658).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),659). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),659).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),660). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),660).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),661). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),661).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),662). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),662).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),663). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),663).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),664). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),664).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),665). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),665).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),666). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),666).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),667). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),667).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),668). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),668).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),669). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),669).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),670). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),670).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),671). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),671).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),672). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),672).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),673). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),673).
:- holdsAt(moving(mwt1gt_5,mwt1gt_6),674). :- holdsAt(moving(mwt1gt_6,mwt1gt_5),674).
:- simpleEvent(T,P,inactive), T>649 , T<673.
:- simpleEvent(T,P,running), T>649 , T<673.

#show simpleEvent/3.