

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,143,337). coords(mwt2gt_1,137,76,337).  visible(mwt2gt_1,337). orientation(mwt2gt_2,145,337). coords(mwt2gt_2,142,87,337).  visible(mwt2gt_2,337).
orientation(mwt2gt_1,143,338). coords(mwt2gt_1,137,76,338).  visible(mwt2gt_1,338). orientation(mwt2gt_2,145,338). coords(mwt2gt_2,142,87,338).  visible(mwt2gt_2,338).
orientation(mwt2gt_1,143,339). coords(mwt2gt_1,137,76,339).  visible(mwt2gt_1,339). orientation(mwt2gt_2,145,339). coords(mwt2gt_2,142,87,339).  visible(mwt2gt_2,339).
orientation(mwt2gt_1,143,340). coords(mwt2gt_1,134,77,340).  visible(mwt2gt_1,340). orientation(mwt2gt_2,141,340). coords(mwt2gt_2,142,86,340).  visible(mwt2gt_2,340).
orientation(mwt2gt_1,143,341). coords(mwt2gt_1,134,77,341).  visible(mwt2gt_1,341). orientation(mwt2gt_2,141,341). coords(mwt2gt_2,142,86,341).  visible(mwt2gt_2,341).
orientation(mwt2gt_1,143,342). coords(mwt2gt_1,134,77,342).  visible(mwt2gt_1,342). orientation(mwt2gt_2,141,342). coords(mwt2gt_2,142,86,342).  visible(mwt2gt_2,342).
orientation(mwt2gt_1,143,343). coords(mwt2gt_1,134,77,343).  visible(mwt2gt_1,343). orientation(mwt2gt_2,141,343). coords(mwt2gt_2,142,86,343).  visible(mwt2gt_2,343).
orientation(mwt2gt_1,143,344). coords(mwt2gt_1,134,77,344).  visible(mwt2gt_1,344). orientation(mwt2gt_2,141,344). coords(mwt2gt_2,142,86,344).  visible(mwt2gt_2,344).
orientation(mwt2gt_1,143,345). coords(mwt2gt_1,134,77,345).  visible(mwt2gt_1,345). orientation(mwt2gt_2,141,345). coords(mwt2gt_2,142,86,345).  visible(mwt2gt_2,345).
orientation(mwt2gt_1,143,346). coords(mwt2gt_1,134,77,346).  visible(mwt2gt_1,346). orientation(mwt2gt_2,141,346). coords(mwt2gt_2,142,86,346).  visible(mwt2gt_2,346).
orientation(mwt2gt_1,143,347). coords(mwt2gt_1,134,77,347).  visible(mwt2gt_1,347). orientation(mwt2gt_2,141,347). coords(mwt2gt_2,142,86,347).  visible(mwt2gt_2,347).
orientation(mwt2gt_1,143,348). coords(mwt2gt_1,134,77,348).  visible(mwt2gt_1,348). orientation(mwt2gt_2,141,348). coords(mwt2gt_2,142,86,348).  visible(mwt2gt_2,348).
orientation(mwt2gt_1,143,349). coords(mwt2gt_1,134,77,349).  visible(mwt2gt_1,349). orientation(mwt2gt_2,141,349). coords(mwt2gt_2,142,86,349).  visible(mwt2gt_2,349).
orientation(mwt2gt_1,143,350). coords(mwt2gt_1,134,77,350).  visible(mwt2gt_1,350). orientation(mwt2gt_2,141,350). coords(mwt2gt_2,142,86,350).  visible(mwt2gt_2,350).
orientation(mwt2gt_1,143,351). coords(mwt2gt_1,134,77,351).  visible(mwt2gt_1,351). orientation(mwt2gt_2,141,351). coords(mwt2gt_2,142,86,351).  visible(mwt2gt_2,351).
orientation(mwt2gt_1,143,352). coords(mwt2gt_1,134,77,352).  visible(mwt2gt_1,352). orientation(mwt2gt_2,141,352). coords(mwt2gt_2,142,86,352).  visible(mwt2gt_2,352).
orientation(mwt2gt_1,143,353). coords(mwt2gt_1,134,77,353).  visible(mwt2gt_1,353). orientation(mwt2gt_2,141,353). coords(mwt2gt_2,142,86,353).  visible(mwt2gt_2,353).
orientation(mwt2gt_1,143,354). coords(mwt2gt_1,134,77,354).  visible(mwt2gt_1,354). orientation(mwt2gt_2,141,354). coords(mwt2gt_2,142,86,354).  visible(mwt2gt_2,354).
orientation(mwt2gt_1,143,355). coords(mwt2gt_1,134,77,355).  visible(mwt2gt_1,355). orientation(mwt2gt_2,141,355). coords(mwt2gt_2,142,86,355).  visible(mwt2gt_2,355).
orientation(mwt2gt_1,143,356). coords(mwt2gt_1,134,77,356).  visible(mwt2gt_1,356). orientation(mwt2gt_2,141,356). coords(mwt2gt_2,142,86,356).  visible(mwt2gt_2,356).
orientation(mwt2gt_1,143,357). coords(mwt2gt_1,134,77,357).  visible(mwt2gt_1,357). orientation(mwt2gt_2,141,357). coords(mwt2gt_2,142,86,357).  visible(mwt2gt_2,357).
orientation(mwt2gt_1,143,358). coords(mwt2gt_1,133,77,358).  visible(mwt2gt_1,358). orientation(mwt2gt_2,141,358). coords(mwt2gt_2,141,86,358).  visible(mwt2gt_2,358).
orientation(mwt2gt_1,143,359). coords(mwt2gt_1,132,77,359).  visible(mwt2gt_1,359). orientation(mwt2gt_2,141,359). coords(mwt2gt_2,141,86,359).  visible(mwt2gt_2,359).
orientation(mwt2gt_1,143,360). coords(mwt2gt_1,132,77,360).  visible(mwt2gt_1,360). orientation(mwt2gt_2,141,360). coords(mwt2gt_2,141,87,360).  visible(mwt2gt_2,360).
orientation(mwt2gt_1,143,361). coords(mwt2gt_1,131,77,361).  visible(mwt2gt_1,361). orientation(mwt2gt_2,143,361). coords(mwt2gt_2,141,88,361).  visible(mwt2gt_2,361).
orientation(mwt2gt_1,143,362). coords(mwt2gt_1,131,77,362).  visible(mwt2gt_1,362). orientation(mwt2gt_2,143,362). coords(mwt2gt_2,141,88,362).  visible(mwt2gt_2,362).
happensAt(disappear(mwt2gt_1),361). happensAt(disappear(mwt2gt_2),361).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=337..361.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=337..361.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),337). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),337).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),338). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),338).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),339). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),339).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),340). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),340).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),341). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),341).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),342). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),342).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),343). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),343).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),344). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),344).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),345). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),345).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),346). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),346).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),347). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),347).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),348). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),348).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),349). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),349).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),350). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),350).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),351). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),351).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),352). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),352).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),353). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),353).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),354). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),354).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),355). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),355).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),356). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),356).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),357). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),357).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),358). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),358).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),359). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),359).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),360). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),360).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),361). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),361).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),362). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),362).
:- simpleEvent(T,P,walking), T>339 , T<361.
:- simpleEvent(T,P,running), T>339 , T<361.

#show simpleEvent/3.