

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

	person(fcgt_1;fcgt_2).
orientation(fcgt_1,147,340). coords(fcgt_1,96,129,340).  visible(fcgt_1,340). orientation(fcgt_2,155,340). coords(fcgt_2,94,117,340).  visible(fcgt_2,340).
orientation(fcgt_1,150,341). coords(fcgt_1,96,129,341).  visible(fcgt_1,341). orientation(fcgt_2,155,341). coords(fcgt_2,94,117,341).  visible(fcgt_2,341).
orientation(fcgt_1,148,342). coords(fcgt_1,96,129,342).  visible(fcgt_1,342). orientation(fcgt_2,155,342). coords(fcgt_2,94,116,342).  visible(fcgt_2,342).
orientation(fcgt_1,146,343). coords(fcgt_1,96,130,343).  visible(fcgt_1,343). orientation(fcgt_2,155,343). coords(fcgt_2,94,115,343).  visible(fcgt_2,343).
orientation(fcgt_1,146,344). coords(fcgt_1,96,130,344).  visible(fcgt_1,344). orientation(fcgt_2,155,344). coords(fcgt_2,94,115,344).  visible(fcgt_2,344).
orientation(fcgt_1,146,345). coords(fcgt_1,97,129,345).  visible(fcgt_1,345). orientation(fcgt_2,155,345). coords(fcgt_2,94,114,345).  visible(fcgt_2,345).
orientation(fcgt_1,146,346). coords(fcgt_1,97,129,346).  visible(fcgt_1,346). orientation(fcgt_2,155,346). coords(fcgt_2,93,114,346).  visible(fcgt_2,346).
orientation(fcgt_1,141,347). coords(fcgt_1,97,127,347).  visible(fcgt_1,347). orientation(fcgt_2,155,347). coords(fcgt_2,94,114,347).  visible(fcgt_2,347).
orientation(fcgt_1,141,348). coords(fcgt_1,97,126,348).  visible(fcgt_1,348). orientation(fcgt_2,155,348). coords(fcgt_2,94,114,348).  visible(fcgt_2,348).
orientation(fcgt_1,141,349). coords(fcgt_1,97,124,349).  visible(fcgt_1,349). orientation(fcgt_2,155,349). coords(fcgt_2,93,113,349).  visible(fcgt_2,349).
orientation(fcgt_1,141,350). coords(fcgt_1,97,123,350).  visible(fcgt_1,350). orientation(fcgt_2,154,350). coords(fcgt_2,93,112,350).  visible(fcgt_2,350).
orientation(fcgt_1,141,351). coords(fcgt_1,96,122,351).  visible(fcgt_1,351). orientation(fcgt_2,149,351). coords(fcgt_2,92,111,351).  visible(fcgt_2,351).
orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). orientation(fcgt_2,147,352). coords(fcgt_2,92,111,352).  visible(fcgt_2,352).
orientation(fcgt_1,141,353). coords(fcgt_1,96,121,353).  visible(fcgt_1,353). orientation(fcgt_2,148,353). coords(fcgt_2,92,110,353).  visible(fcgt_2,353).
orientation(fcgt_1,141,354). coords(fcgt_1,96,121,354).  visible(fcgt_1,354). orientation(fcgt_2,146,354). coords(fcgt_2,92,110,354).  visible(fcgt_2,354).
orientation(fcgt_1,141,355). coords(fcgt_1,96,121,355).  visible(fcgt_1,355). orientation(fcgt_2,147,355). coords(fcgt_2,92,109,355).  visible(fcgt_2,355).
orientation(fcgt_1,141,356). coords(fcgt_1,96,120,356).  visible(fcgt_1,356). orientation(fcgt_2,142,356). coords(fcgt_2,92,107,356).  visible(fcgt_2,356).
orientation(fcgt_1,141,357). coords(fcgt_1,96,121,357).  visible(fcgt_1,357). orientation(fcgt_2,142,357). coords(fcgt_2,92,106,357).  visible(fcgt_2,357).
orientation(fcgt_1,141,358). coords(fcgt_1,96,121,358).  visible(fcgt_1,358). orientation(fcgt_2,142,358). coords(fcgt_2,92,104,358).  visible(fcgt_2,358).
orientation(fcgt_1,141,359). coords(fcgt_1,96,122,359).  visible(fcgt_1,359). orientation(fcgt_2,142,359). coords(fcgt_2,91,104,359).  visible(fcgt_2,359).
orientation(fcgt_1,141,360). coords(fcgt_1,95,122,360).  visible(fcgt_1,360). orientation(fcgt_2,142,360). coords(fcgt_2,91,103,360).  visible(fcgt_2,360).
orientation(fcgt_1,141,361). coords(fcgt_1,95,121,361).  visible(fcgt_1,361). orientation(fcgt_2,142,361). coords(fcgt_2,91,102,361).  visible(fcgt_2,361).
orientation(fcgt_1,141,362). coords(fcgt_1,95,120,362).  visible(fcgt_1,362). orientation(fcgt_2,142,362). coords(fcgt_2,91,102,362).  visible(fcgt_2,362).
orientation(fcgt_1,141,363). coords(fcgt_1,95,117,363).  visible(fcgt_1,363). orientation(fcgt_2,142,363). coords(fcgt_2,91,102,363).  visible(fcgt_2,363).
orientation(fcgt_1,141,364). coords(fcgt_1,95,115,364).  visible(fcgt_1,364). orientation(fcgt_2,142,364). coords(fcgt_2,91,102,364).  visible(fcgt_2,364).
orientation(fcgt_1,141,365). coords(fcgt_1,95,115,365).  visible(fcgt_1,365). orientation(fcgt_2,142,365). coords(fcgt_2,91,102,365).  visible(fcgt_2,365).
happensAt(disappear(fcgt_1),364). happensAt(disappear(fcgt_2),364).
1{simpleEvent(T,fcgt_1,active); simpleEvent(T,fcgt_1,inactive); simpleEvent(T,fcgt_1,walking); simpleEvent(T,fcgt_1,running)}1 :- T=340..364.
1{simpleEvent(T,fcgt_2,active); simpleEvent(T,fcgt_2,inactive); simpleEvent(T,fcgt_2,walking); simpleEvent(T,fcgt_2,running)}1 :- T=340..364.
:- holdsAt(meeting(fcgt_1,fcgt_2),340). :- holdsAt(meeting(fcgt_2,fcgt_1),340).
:- not holdsAt(meeting(fcgt_1,fcgt_2),341). :- not holdsAt(meeting(fcgt_2,fcgt_1),341).
:- not holdsAt(meeting(fcgt_1,fcgt_2),342). :- not holdsAt(meeting(fcgt_2,fcgt_1),342).
:- not holdsAt(meeting(fcgt_1,fcgt_2),343). :- not holdsAt(meeting(fcgt_2,fcgt_1),343).
:- not holdsAt(meeting(fcgt_1,fcgt_2),344). :- not holdsAt(meeting(fcgt_2,fcgt_1),344).
:- not holdsAt(meeting(fcgt_1,fcgt_2),345). :- not holdsAt(meeting(fcgt_2,fcgt_1),345).
:- not holdsAt(meeting(fcgt_1,fcgt_2),346). :- not holdsAt(meeting(fcgt_2,fcgt_1),346).
:- not holdsAt(meeting(fcgt_1,fcgt_2),347). :- not holdsAt(meeting(fcgt_2,fcgt_1),347).
:- not holdsAt(meeting(fcgt_1,fcgt_2),348). :- not holdsAt(meeting(fcgt_2,fcgt_1),348).
:- not holdsAt(meeting(fcgt_1,fcgt_2),349). :- not holdsAt(meeting(fcgt_2,fcgt_1),349).
:- not holdsAt(meeting(fcgt_1,fcgt_2),350). :- not holdsAt(meeting(fcgt_2,fcgt_1),350).
:- not holdsAt(meeting(fcgt_1,fcgt_2),351). :- not holdsAt(meeting(fcgt_2,fcgt_1),351).
:- not holdsAt(meeting(fcgt_1,fcgt_2),352). :- not holdsAt(meeting(fcgt_2,fcgt_1),352).
:- not holdsAt(meeting(fcgt_1,fcgt_2),353). :- not holdsAt(meeting(fcgt_2,fcgt_1),353).
:- not holdsAt(meeting(fcgt_1,fcgt_2),354). :- not holdsAt(meeting(fcgt_2,fcgt_1),354).
:- not holdsAt(meeting(fcgt_1,fcgt_2),355). :- not holdsAt(meeting(fcgt_2,fcgt_1),355).
:- not holdsAt(meeting(fcgt_1,fcgt_2),356). :- not holdsAt(meeting(fcgt_2,fcgt_1),356).
:- not holdsAt(meeting(fcgt_1,fcgt_2),357). :- not holdsAt(meeting(fcgt_2,fcgt_1),357).
:- not holdsAt(meeting(fcgt_1,fcgt_2),358). :- not holdsAt(meeting(fcgt_2,fcgt_1),358).
:- not holdsAt(meeting(fcgt_1,fcgt_2),359). :- not holdsAt(meeting(fcgt_2,fcgt_1),359).
:- not holdsAt(meeting(fcgt_1,fcgt_2),360). :- not holdsAt(meeting(fcgt_2,fcgt_1),360).
:- not holdsAt(meeting(fcgt_1,fcgt_2),361). :- not holdsAt(meeting(fcgt_2,fcgt_1),361).
:- not holdsAt(meeting(fcgt_1,fcgt_2),362). :- not holdsAt(meeting(fcgt_2,fcgt_1),362).
:- not holdsAt(meeting(fcgt_1,fcgt_2),363). :- not holdsAt(meeting(fcgt_2,fcgt_1),363).
:- not holdsAt(meeting(fcgt_1,fcgt_2),364). :- not holdsAt(meeting(fcgt_2,fcgt_1),364).
:- holdsAt(meeting(fcgt_1,fcgt_2),365). :- holdsAt(meeting(fcgt_2,fcgt_1),365).
:- simpleEvent(T,P,walking), T>340 , T<364.
:- simpleEvent(T,P,running), T>340 , T<364.

#show simpleEvent/3.