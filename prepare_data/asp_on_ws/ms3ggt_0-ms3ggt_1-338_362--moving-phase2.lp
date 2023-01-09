

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

	person(ms3ggt_0;ms3ggt_1).
orientation(ms3ggt_0,163,338). coords(ms3ggt_0,107,145,338).  visible(ms3ggt_0,338). orientation(ms3ggt_1,156,338). coords(ms3ggt_1,104,166,338).  visible(ms3ggt_1,338).
orientation(ms3ggt_0,156,339). coords(ms3ggt_0,107,144,339).  visible(ms3ggt_0,339). orientation(ms3ggt_1,156,339). coords(ms3ggt_1,104,166,339).  visible(ms3ggt_1,339).
orientation(ms3ggt_0,156,340). coords(ms3ggt_0,108,144,340).  visible(ms3ggt_0,340). orientation(ms3ggt_1,156,340). coords(ms3ggt_1,105,166,340).  visible(ms3ggt_1,340).
orientation(ms3ggt_0,156,341). coords(ms3ggt_0,108,144,341).  visible(ms3ggt_0,341). orientation(ms3ggt_1,156,341). coords(ms3ggt_1,105,166,341).  visible(ms3ggt_1,341).
orientation(ms3ggt_0,156,342). coords(ms3ggt_0,109,144,342).  visible(ms3ggt_0,342). orientation(ms3ggt_1,156,342). coords(ms3ggt_1,105,166,342).  visible(ms3ggt_1,342).
orientation(ms3ggt_0,156,343). coords(ms3ggt_0,109,144,343).  visible(ms3ggt_0,343). orientation(ms3ggt_1,156,343). coords(ms3ggt_1,105,166,343).  visible(ms3ggt_1,343).
orientation(ms3ggt_0,158,344). coords(ms3ggt_0,112,144,344).  visible(ms3ggt_0,344). orientation(ms3ggt_1,156,344). coords(ms3ggt_1,106,166,344).  visible(ms3ggt_1,344).
orientation(ms3ggt_0,152,345). coords(ms3ggt_0,114,144,345).  visible(ms3ggt_0,345). orientation(ms3ggt_1,156,345). coords(ms3ggt_1,108,165,345).  visible(ms3ggt_1,345).
orientation(ms3ggt_0,152,346). coords(ms3ggt_0,116,144,346).  visible(ms3ggt_0,346). orientation(ms3ggt_1,156,346). coords(ms3ggt_1,110,165,346).  visible(ms3ggt_1,346).
orientation(ms3ggt_0,152,347). coords(ms3ggt_0,117,144,347).  visible(ms3ggt_0,347). orientation(ms3ggt_1,156,347). coords(ms3ggt_1,111,164,347).  visible(ms3ggt_1,347).
orientation(ms3ggt_0,157,348). coords(ms3ggt_0,118,144,348).  visible(ms3ggt_0,348). orientation(ms3ggt_1,156,348). coords(ms3ggt_1,112,164,348).  visible(ms3ggt_1,348).
orientation(ms3ggt_0,158,349). coords(ms3ggt_0,119,144,349).  visible(ms3ggt_0,349). orientation(ms3ggt_1,156,349). coords(ms3ggt_1,113,164,349).  visible(ms3ggt_1,349).
orientation(ms3ggt_0,158,350). coords(ms3ggt_0,120,145,350).  visible(ms3ggt_0,350). orientation(ms3ggt_1,156,350). coords(ms3ggt_1,114,164,350).  visible(ms3ggt_1,350).
orientation(ms3ggt_0,160,351). coords(ms3ggt_0,121,145,351).  visible(ms3ggt_0,351). orientation(ms3ggt_1,156,351). coords(ms3ggt_1,115,164,351).  visible(ms3ggt_1,351).
orientation(ms3ggt_0,160,352). coords(ms3ggt_0,121,145,352).  visible(ms3ggt_0,352). orientation(ms3ggt_1,156,352). coords(ms3ggt_1,115,163,352).  visible(ms3ggt_1,352).
orientation(ms3ggt_0,160,353). coords(ms3ggt_0,121,145,353).  visible(ms3ggt_0,353). orientation(ms3ggt_1,156,353). coords(ms3ggt_1,115,163,353).  visible(ms3ggt_1,353).
orientation(ms3ggt_0,160,354). coords(ms3ggt_0,121,145,354).  visible(ms3ggt_0,354). orientation(ms3ggt_1,156,354). coords(ms3ggt_1,115,163,354).  visible(ms3ggt_1,354).
orientation(ms3ggt_0,157,355). coords(ms3ggt_0,121,144,355).  visible(ms3ggt_0,355). orientation(ms3ggt_1,156,355). coords(ms3ggt_1,116,162,355).  visible(ms3ggt_1,355).
orientation(ms3ggt_0,157,356). coords(ms3ggt_0,122,144,356).  visible(ms3ggt_0,356). orientation(ms3ggt_1,156,356). coords(ms3ggt_1,116,162,356).  visible(ms3ggt_1,356).
orientation(ms3ggt_0,157,357). coords(ms3ggt_0,123,144,357).  visible(ms3ggt_0,357). orientation(ms3ggt_1,156,357). coords(ms3ggt_1,117,161,357).  visible(ms3ggt_1,357).
orientation(ms3ggt_0,157,358). coords(ms3ggt_0,125,144,358).  visible(ms3ggt_0,358). orientation(ms3ggt_1,156,358). coords(ms3ggt_1,117,161,358).  visible(ms3ggt_1,358).
orientation(ms3ggt_0,157,359). coords(ms3ggt_0,127,144,359).  visible(ms3ggt_0,359). orientation(ms3ggt_1,156,359). coords(ms3ggt_1,117,161,359).  visible(ms3ggt_1,359).
orientation(ms3ggt_0,161,360). coords(ms3ggt_0,129,144,360).  visible(ms3ggt_0,360). orientation(ms3ggt_1,156,360). coords(ms3ggt_1,117,161,360).  visible(ms3ggt_1,360).
orientation(ms3ggt_0,161,361). coords(ms3ggt_0,130,143,361).  visible(ms3ggt_0,361). orientation(ms3ggt_1,156,361). coords(ms3ggt_1,117,159,361).  visible(ms3ggt_1,361).
orientation(ms3ggt_0,161,362). coords(ms3ggt_0,131,143,362).  visible(ms3ggt_0,362). orientation(ms3ggt_1,156,362). coords(ms3ggt_1,118,159,362).  visible(ms3ggt_1,362).
orientation(ms3ggt_0,161,363). coords(ms3ggt_0,131,143,363).  visible(ms3ggt_0,363). orientation(ms3ggt_1,156,363). coords(ms3ggt_1,118,159,363).  visible(ms3ggt_1,363).
happensAt(disappear(ms3ggt_0),362). happensAt(disappear(ms3ggt_1),362).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=338..362.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=338..362.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),338). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),338).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),339). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),339).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),340). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),340).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),341). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),341).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),342). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),342).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),343). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),343).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),344). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),344).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),345). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),345).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),346). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),346).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),347). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),347).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),348). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),348).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),349). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),349).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),350). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),350).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),351). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),351).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),352). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),352).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),353). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),353).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),354). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),354).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),355). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),355).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),356). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),356).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),357). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),357).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),358). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),358).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),359). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),359).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),360). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),360).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),361). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),361).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),362). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),362).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),363). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),363).
:- simpleEvent(T,P,inactive), T>338 , T<362.
:- simpleEvent(T,P,running), T>338 , T<362.

#show simpleEvent/3.