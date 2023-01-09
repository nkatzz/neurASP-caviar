

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

	person(fra2gt_3;fra2gt_4).
orientation(fra2gt_3,138,346). coords(fra2gt_3,90,111,346).  visible(fra2gt_3,346). orientation(fra2gt_4,155,346). coords(fra2gt_4,98,119,346).  visible(fra2gt_4,346).
orientation(fra2gt_3,138,347). coords(fra2gt_3,90,111,347).  visible(fra2gt_3,347). orientation(fra2gt_4,155,347). coords(fra2gt_4,97,118,347).  visible(fra2gt_4,347).
orientation(fra2gt_3,138,348). coords(fra2gt_3,89,111,348).  visible(fra2gt_3,348). orientation(fra2gt_4,155,348). coords(fra2gt_4,95,117,348).  visible(fra2gt_4,348).
orientation(fra2gt_3,138,349). coords(fra2gt_3,89,111,349).  visible(fra2gt_3,349). orientation(fra2gt_4,155,349). coords(fra2gt_4,95,116,349).  visible(fra2gt_4,349).
orientation(fra2gt_3,138,350). coords(fra2gt_3,89,111,350).  visible(fra2gt_3,350). orientation(fra2gt_4,155,350). coords(fra2gt_4,95,116,350).  visible(fra2gt_4,350).
orientation(fra2gt_3,138,351). coords(fra2gt_3,88,111,351).  visible(fra2gt_3,351). orientation(fra2gt_4,155,351). coords(fra2gt_4,96,116,351).  visible(fra2gt_4,351).
orientation(fra2gt_3,138,352). coords(fra2gt_3,88,112,352).  visible(fra2gt_3,352). orientation(fra2gt_4,155,352). coords(fra2gt_4,95,116,352).  visible(fra2gt_4,352).
orientation(fra2gt_3,138,353). coords(fra2gt_3,85,112,353).  visible(fra2gt_3,353). orientation(fra2gt_4,155,353). coords(fra2gt_4,95,116,353).  visible(fra2gt_4,353).
orientation(fra2gt_3,138,354). coords(fra2gt_3,85,113,354).  visible(fra2gt_3,354). orientation(fra2gt_4,155,354). coords(fra2gt_4,97,115,354).  visible(fra2gt_4,354).
orientation(fra2gt_3,138,355). coords(fra2gt_3,83,114,355).  visible(fra2gt_3,355). orientation(fra2gt_4,155,355). coords(fra2gt_4,97,116,355).  visible(fra2gt_4,355).
orientation(fra2gt_3,138,356). coords(fra2gt_3,82,114,356).  visible(fra2gt_3,356). orientation(fra2gt_4,155,356). coords(fra2gt_4,95,116,356).  visible(fra2gt_4,356).
orientation(fra2gt_3,138,357). coords(fra2gt_3,83,114,357).  visible(fra2gt_3,357). orientation(fra2gt_4,155,357). coords(fra2gt_4,95,118,357).  visible(fra2gt_4,357).
orientation(fra2gt_3,138,358). coords(fra2gt_3,83,114,358).  visible(fra2gt_3,358). orientation(fra2gt_4,155,358). coords(fra2gt_4,95,117,358).  visible(fra2gt_4,358).
orientation(fra2gt_3,138,359). coords(fra2gt_3,83,115,359).  visible(fra2gt_3,359). orientation(fra2gt_4,155,359). coords(fra2gt_4,94,118,359).  visible(fra2gt_4,359).
orientation(fra2gt_3,138,360). coords(fra2gt_3,83,116,360).  visible(fra2gt_3,360). orientation(fra2gt_4,155,360). coords(fra2gt_4,94,118,360).  visible(fra2gt_4,360).
orientation(fra2gt_3,138,361). coords(fra2gt_3,83,117,361).  visible(fra2gt_3,361). orientation(fra2gt_4,155,361). coords(fra2gt_4,93,117,361).  visible(fra2gt_4,361).
orientation(fra2gt_3,138,362). coords(fra2gt_3,82,118,362).  visible(fra2gt_3,362). orientation(fra2gt_4,155,362). coords(fra2gt_4,92,118,362).  visible(fra2gt_4,362).
orientation(fra2gt_3,138,363). coords(fra2gt_3,82,118,363).  visible(fra2gt_3,363). orientation(fra2gt_4,155,363). coords(fra2gt_4,91,119,363).  visible(fra2gt_4,363).
orientation(fra2gt_3,138,364). coords(fra2gt_3,82,118,364).  visible(fra2gt_3,364). orientation(fra2gt_4,155,364). coords(fra2gt_4,90,119,364).  visible(fra2gt_4,364).
orientation(fra2gt_3,138,365). coords(fra2gt_3,76,118,365).  visible(fra2gt_3,365). orientation(fra2gt_4,155,365). coords(fra2gt_4,89,119,365).  visible(fra2gt_4,365).
orientation(fra2gt_3,138,366). coords(fra2gt_3,79,118,366).  visible(fra2gt_3,366). orientation(fra2gt_4,155,366). coords(fra2gt_4,87,117,366).  visible(fra2gt_4,366).
orientation(fra2gt_3,138,367). coords(fra2gt_3,79,120,367).  visible(fra2gt_3,367). orientation(fra2gt_4,155,367). coords(fra2gt_4,86,117,367).  visible(fra2gt_4,367).
orientation(fra2gt_3,138,368). coords(fra2gt_3,77,120,368).  visible(fra2gt_3,368). orientation(fra2gt_4,155,368). coords(fra2gt_4,84,117,368).  visible(fra2gt_4,368).
orientation(fra2gt_3,138,369). coords(fra2gt_3,77,121,369).  visible(fra2gt_3,369). orientation(fra2gt_4,155,369). coords(fra2gt_4,83,119,369).  visible(fra2gt_4,369).
orientation(fra2gt_3,138,370). coords(fra2gt_3,77,121,370).  visible(fra2gt_3,370). orientation(fra2gt_4,155,370). coords(fra2gt_4,83,120,370).  visible(fra2gt_4,370).
orientation(fra2gt_3,138,371). coords(fra2gt_3,77,121,371).  visible(fra2gt_3,371). orientation(fra2gt_4,155,371). coords(fra2gt_4,83,120,371).  visible(fra2gt_4,371).
happensAt(disappear(fra2gt_3),370). happensAt(disappear(fra2gt_4),370).
1{simpleEvent(T,fra2gt_3,active); simpleEvent(T,fra2gt_3,inactive); simpleEvent(T,fra2gt_3,walking); simpleEvent(T,fra2gt_3,running)}1 :- T=346..370.
1{simpleEvent(T,fra2gt_4,active); simpleEvent(T,fra2gt_4,inactive); simpleEvent(T,fra2gt_4,walking); simpleEvent(T,fra2gt_4,running)}1 :- T=346..370.
:- holdsAt(moving(fra2gt_3,fra2gt_4),346). :- holdsAt(moving(fra2gt_4,fra2gt_3),346).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),347). :- not holdsAt(moving(fra2gt_4,fra2gt_3),347).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),348). :- not holdsAt(moving(fra2gt_4,fra2gt_3),348).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),349). :- not holdsAt(moving(fra2gt_4,fra2gt_3),349).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),350). :- not holdsAt(moving(fra2gt_4,fra2gt_3),350).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),351). :- not holdsAt(moving(fra2gt_4,fra2gt_3),351).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),352). :- not holdsAt(moving(fra2gt_4,fra2gt_3),352).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),353). :- not holdsAt(moving(fra2gt_4,fra2gt_3),353).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),354). :- not holdsAt(moving(fra2gt_4,fra2gt_3),354).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),355). :- not holdsAt(moving(fra2gt_4,fra2gt_3),355).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),356). :- not holdsAt(moving(fra2gt_4,fra2gt_3),356).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),357). :- not holdsAt(moving(fra2gt_4,fra2gt_3),357).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),358). :- not holdsAt(moving(fra2gt_4,fra2gt_3),358).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),359). :- not holdsAt(moving(fra2gt_4,fra2gt_3),359).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),360). :- not holdsAt(moving(fra2gt_4,fra2gt_3),360).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),361). :- not holdsAt(moving(fra2gt_4,fra2gt_3),361).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),362). :- not holdsAt(moving(fra2gt_4,fra2gt_3),362).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),363). :- not holdsAt(moving(fra2gt_4,fra2gt_3),363).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),364). :- not holdsAt(moving(fra2gt_4,fra2gt_3),364).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),365). :- not holdsAt(moving(fra2gt_4,fra2gt_3),365).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),366). :- not holdsAt(moving(fra2gt_4,fra2gt_3),366).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),367). :- not holdsAt(moving(fra2gt_4,fra2gt_3),367).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),368). :- not holdsAt(moving(fra2gt_4,fra2gt_3),368).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),369). :- not holdsAt(moving(fra2gt_4,fra2gt_3),369).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),370). :- not holdsAt(moving(fra2gt_4,fra2gt_3),370).
:- holdsAt(moving(fra2gt_3,fra2gt_4),371). :- holdsAt(moving(fra2gt_4,fra2gt_3),371).
:- simpleEvent(T,P,inactive), T>346 , T<370.
:- simpleEvent(T,P,running), T>346 , T<370.

#show simpleEvent/3.