

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

	person(fcgt_3;fcgt_4).
orientation(fcgt_3,164,336). coords(fcgt_3,98,187,336).  visible(fcgt_3,336). orientation(fcgt_4,166,336). coords(fcgt_4,82,206,336).  visible(fcgt_4,336).
orientation(fcgt_3,164,337). coords(fcgt_3,98,187,337).  visible(fcgt_3,337). orientation(fcgt_4,166,337). coords(fcgt_4,82,206,337).  visible(fcgt_4,337).
orientation(fcgt_3,164,338). coords(fcgt_3,98,187,338).  visible(fcgt_3,338). orientation(fcgt_4,166,338). coords(fcgt_4,82,206,338).  visible(fcgt_4,338).
orientation(fcgt_3,164,339). coords(fcgt_3,98,187,339).  visible(fcgt_3,339). orientation(fcgt_4,166,339). coords(fcgt_4,82,206,339).  visible(fcgt_4,339).
orientation(fcgt_3,164,340). coords(fcgt_3,98,187,340).  visible(fcgt_3,340). orientation(fcgt_4,166,340). coords(fcgt_4,82,206,340).  visible(fcgt_4,340).
orientation(fcgt_3,164,341). coords(fcgt_3,98,187,341).  visible(fcgt_3,341). orientation(fcgt_4,166,341). coords(fcgt_4,82,206,341).  visible(fcgt_4,341).
orientation(fcgt_3,164,342). coords(fcgt_3,98,187,342).  visible(fcgt_3,342). orientation(fcgt_4,166,342). coords(fcgt_4,82,206,342).  visible(fcgt_4,342).
orientation(fcgt_3,164,343). coords(fcgt_3,98,187,343).  visible(fcgt_3,343). orientation(fcgt_4,166,343). coords(fcgt_4,82,206,343).  visible(fcgt_4,343).
orientation(fcgt_3,164,344). coords(fcgt_3,98,187,344).  visible(fcgt_3,344). orientation(fcgt_4,166,344). coords(fcgt_4,82,206,344).  visible(fcgt_4,344).
orientation(fcgt_3,164,345). coords(fcgt_3,98,187,345).  visible(fcgt_3,345). orientation(fcgt_4,166,345). coords(fcgt_4,82,206,345).  visible(fcgt_4,345).
orientation(fcgt_3,164,346). coords(fcgt_3,98,187,346).  visible(fcgt_3,346). orientation(fcgt_4,166,346). coords(fcgt_4,82,206,346).  visible(fcgt_4,346).
orientation(fcgt_3,164,347). coords(fcgt_3,98,187,347).  visible(fcgt_3,347). orientation(fcgt_4,166,347). coords(fcgt_4,82,206,347).  visible(fcgt_4,347).
orientation(fcgt_3,164,348). coords(fcgt_3,98,187,348).  visible(fcgt_3,348). orientation(fcgt_4,166,348). coords(fcgt_4,82,206,348).  visible(fcgt_4,348).
orientation(fcgt_3,164,349). coords(fcgt_3,98,187,349).  visible(fcgt_3,349). orientation(fcgt_4,166,349). coords(fcgt_4,82,206,349).  visible(fcgt_4,349).
orientation(fcgt_3,164,350). coords(fcgt_3,98,187,350).  visible(fcgt_3,350). orientation(fcgt_4,166,350). coords(fcgt_4,82,206,350).  visible(fcgt_4,350).
orientation(fcgt_3,164,351). coords(fcgt_3,99,187,351).  visible(fcgt_3,351). orientation(fcgt_4,166,351). coords(fcgt_4,82,206,351).  visible(fcgt_4,351).
orientation(fcgt_3,164,352). coords(fcgt_3,99,187,352).  visible(fcgt_3,352). orientation(fcgt_4,166,352). coords(fcgt_4,82,206,352).  visible(fcgt_4,352).
orientation(fcgt_3,164,353). coords(fcgt_3,99,187,353).  visible(fcgt_3,353). orientation(fcgt_4,166,353). coords(fcgt_4,82,206,353).  visible(fcgt_4,353).
orientation(fcgt_3,164,354). coords(fcgt_3,99,187,354).  visible(fcgt_3,354). orientation(fcgt_4,166,354). coords(fcgt_4,82,206,354).  visible(fcgt_4,354).
orientation(fcgt_3,164,355). coords(fcgt_3,99,187,355).  visible(fcgt_3,355). orientation(fcgt_4,166,355). coords(fcgt_4,82,206,355).  visible(fcgt_4,355).
orientation(fcgt_3,164,356). coords(fcgt_3,99,187,356).  visible(fcgt_3,356). orientation(fcgt_4,166,356). coords(fcgt_4,82,206,356).  visible(fcgt_4,356).
orientation(fcgt_3,164,357). coords(fcgt_3,99,187,357).  visible(fcgt_3,357). orientation(fcgt_4,166,357). coords(fcgt_4,82,206,357).  visible(fcgt_4,357).
orientation(fcgt_3,164,358). coords(fcgt_3,99,187,358).  visible(fcgt_3,358). orientation(fcgt_4,166,358). coords(fcgt_4,82,206,358).  visible(fcgt_4,358).
orientation(fcgt_3,164,359). coords(fcgt_3,99,187,359).  visible(fcgt_3,359). orientation(fcgt_4,166,359). coords(fcgt_4,82,206,359).  visible(fcgt_4,359).
orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). orientation(fcgt_4,166,360). coords(fcgt_4,82,206,360).  visible(fcgt_4,360).
orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). orientation(fcgt_4,166,361). coords(fcgt_4,82,206,361).  visible(fcgt_4,361).
happensAt(disappear(fcgt_3),360). happensAt(disappear(fcgt_4),360).
1{simpleEvent(T,fcgt_3,active); simpleEvent(T,fcgt_3,inactive); simpleEvent(T,fcgt_3,walking); simpleEvent(T,fcgt_3,running)}1 :- T=336..360.
1{simpleEvent(T,fcgt_4,active); simpleEvent(T,fcgt_4,inactive); simpleEvent(T,fcgt_4,walking); simpleEvent(T,fcgt_4,running)}1 :- T=336..360.
:- holdsAt(meeting(fcgt_3,fcgt_4),336). :- holdsAt(meeting(fcgt_4,fcgt_3),336).
:- not holdsAt(meeting(fcgt_3,fcgt_4),337). :- not holdsAt(meeting(fcgt_4,fcgt_3),337).
:- not holdsAt(meeting(fcgt_3,fcgt_4),338). :- not holdsAt(meeting(fcgt_4,fcgt_3),338).
:- not holdsAt(meeting(fcgt_3,fcgt_4),339). :- not holdsAt(meeting(fcgt_4,fcgt_3),339).
:- not holdsAt(meeting(fcgt_3,fcgt_4),340). :- not holdsAt(meeting(fcgt_4,fcgt_3),340).
:- not holdsAt(meeting(fcgt_3,fcgt_4),341). :- not holdsAt(meeting(fcgt_4,fcgt_3),341).
:- not holdsAt(meeting(fcgt_3,fcgt_4),342). :- not holdsAt(meeting(fcgt_4,fcgt_3),342).
:- not holdsAt(meeting(fcgt_3,fcgt_4),343). :- not holdsAt(meeting(fcgt_4,fcgt_3),343).
:- not holdsAt(meeting(fcgt_3,fcgt_4),344). :- not holdsAt(meeting(fcgt_4,fcgt_3),344).
:- not holdsAt(meeting(fcgt_3,fcgt_4),345). :- not holdsAt(meeting(fcgt_4,fcgt_3),345).
:- not holdsAt(meeting(fcgt_3,fcgt_4),346). :- not holdsAt(meeting(fcgt_4,fcgt_3),346).
:- not holdsAt(meeting(fcgt_3,fcgt_4),347). :- not holdsAt(meeting(fcgt_4,fcgt_3),347).
:- not holdsAt(meeting(fcgt_3,fcgt_4),348). :- not holdsAt(meeting(fcgt_4,fcgt_3),348).
:- not holdsAt(meeting(fcgt_3,fcgt_4),349). :- not holdsAt(meeting(fcgt_4,fcgt_3),349).
:- not holdsAt(meeting(fcgt_3,fcgt_4),350). :- not holdsAt(meeting(fcgt_4,fcgt_3),350).
:- not holdsAt(meeting(fcgt_3,fcgt_4),351). :- not holdsAt(meeting(fcgt_4,fcgt_3),351).
:- not holdsAt(meeting(fcgt_3,fcgt_4),352). :- not holdsAt(meeting(fcgt_4,fcgt_3),352).
:- not holdsAt(meeting(fcgt_3,fcgt_4),353). :- not holdsAt(meeting(fcgt_4,fcgt_3),353).
:- not holdsAt(meeting(fcgt_3,fcgt_4),354). :- not holdsAt(meeting(fcgt_4,fcgt_3),354).
:- not holdsAt(meeting(fcgt_3,fcgt_4),355). :- not holdsAt(meeting(fcgt_4,fcgt_3),355).
:- not holdsAt(meeting(fcgt_3,fcgt_4),356). :- not holdsAt(meeting(fcgt_4,fcgt_3),356).
:- not holdsAt(meeting(fcgt_3,fcgt_4),357). :- not holdsAt(meeting(fcgt_4,fcgt_3),357).
:- not holdsAt(meeting(fcgt_3,fcgt_4),358). :- not holdsAt(meeting(fcgt_4,fcgt_3),358).
:- not holdsAt(meeting(fcgt_3,fcgt_4),359). :- not holdsAt(meeting(fcgt_4,fcgt_3),359).
:- not holdsAt(meeting(fcgt_3,fcgt_4),360). :- not holdsAt(meeting(fcgt_4,fcgt_3),360).
:- holdsAt(meeting(fcgt_3,fcgt_4),361). :- holdsAt(meeting(fcgt_4,fcgt_3),361).
:- simpleEvent(T,P,walking), T>336 , T<360.
:- simpleEvent(T,P,running), T>336 , T<360.

#show simpleEvent/3.