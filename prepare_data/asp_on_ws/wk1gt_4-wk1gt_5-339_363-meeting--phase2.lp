

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,160,339). coords(wk1gt_4,52,176,339).  visible(wk1gt_4,339). orientation(wk1gt_5,162,339). coords(wk1gt_5,50,163,339).  visible(wk1gt_5,339).
orientation(wk1gt_4,158,340). coords(wk1gt_4,52,176,340).  visible(wk1gt_4,340). orientation(wk1gt_5,162,340). coords(wk1gt_5,50,163,340).  visible(wk1gt_5,340).
orientation(wk1gt_4,162,341). coords(wk1gt_4,52,176,341).  visible(wk1gt_4,341). orientation(wk1gt_5,162,341). coords(wk1gt_5,50,163,341).  visible(wk1gt_5,341).
orientation(wk1gt_4,165,342). coords(wk1gt_4,52,176,342).  visible(wk1gt_4,342). orientation(wk1gt_5,162,342). coords(wk1gt_5,50,163,342).  visible(wk1gt_5,342).
orientation(wk1gt_4,162,343). coords(wk1gt_4,52,176,343).  visible(wk1gt_4,343). orientation(wk1gt_5,162,343). coords(wk1gt_5,50,163,343).  visible(wk1gt_5,343).
orientation(wk1gt_4,167,344). coords(wk1gt_4,52,176,344).  visible(wk1gt_4,344). orientation(wk1gt_5,162,344). coords(wk1gt_5,50,163,344).  visible(wk1gt_5,344).
orientation(wk1gt_4,164,345). coords(wk1gt_4,52,176,345).  visible(wk1gt_4,345). orientation(wk1gt_5,162,345). coords(wk1gt_5,50,163,345).  visible(wk1gt_5,345).
orientation(wk1gt_4,168,346). coords(wk1gt_4,52,176,346).  visible(wk1gt_4,346). orientation(wk1gt_5,162,346). coords(wk1gt_5,50,163,346).  visible(wk1gt_5,346).
orientation(wk1gt_4,168,347). coords(wk1gt_4,52,176,347).  visible(wk1gt_4,347). orientation(wk1gt_5,162,347). coords(wk1gt_5,50,163,347).  visible(wk1gt_5,347).
orientation(wk1gt_4,166,348). coords(wk1gt_4,52,176,348).  visible(wk1gt_4,348). orientation(wk1gt_5,162,348). coords(wk1gt_5,50,163,348).  visible(wk1gt_5,348).
orientation(wk1gt_4,165,349). coords(wk1gt_4,52,176,349).  visible(wk1gt_4,349). orientation(wk1gt_5,162,349). coords(wk1gt_5,50,163,349).  visible(wk1gt_5,349).
orientation(wk1gt_4,165,350). coords(wk1gt_4,52,176,350).  visible(wk1gt_4,350). orientation(wk1gt_5,162,350). coords(wk1gt_5,50,163,350).  visible(wk1gt_5,350).
orientation(wk1gt_4,167,351). coords(wk1gt_4,52,176,351).  visible(wk1gt_4,351). orientation(wk1gt_5,162,351). coords(wk1gt_5,50,163,351).  visible(wk1gt_5,351).
orientation(wk1gt_4,162,352). coords(wk1gt_4,52,176,352).  visible(wk1gt_4,352). orientation(wk1gt_5,162,352). coords(wk1gt_5,50,163,352).  visible(wk1gt_5,352).
orientation(wk1gt_4,159,353). coords(wk1gt_4,52,176,353).  visible(wk1gt_4,353). orientation(wk1gt_5,162,353). coords(wk1gt_5,50,163,353).  visible(wk1gt_5,353).
orientation(wk1gt_4,162,354). coords(wk1gt_4,52,176,354).  visible(wk1gt_4,354). orientation(wk1gt_5,162,354). coords(wk1gt_5,50,163,354).  visible(wk1gt_5,354).
orientation(wk1gt_4,165,355). coords(wk1gt_4,52,176,355).  visible(wk1gt_4,355). orientation(wk1gt_5,162,355). coords(wk1gt_5,50,163,355).  visible(wk1gt_5,355).
orientation(wk1gt_4,158,356). coords(wk1gt_4,52,176,356).  visible(wk1gt_4,356). orientation(wk1gt_5,162,356). coords(wk1gt_5,50,163,356).  visible(wk1gt_5,356).
orientation(wk1gt_4,157,357). coords(wk1gt_4,52,176,357).  visible(wk1gt_4,357). orientation(wk1gt_5,162,357). coords(wk1gt_5,50,163,357).  visible(wk1gt_5,357).
orientation(wk1gt_4,161,358). coords(wk1gt_4,52,176,358).  visible(wk1gt_4,358). orientation(wk1gt_5,162,358). coords(wk1gt_5,50,163,358).  visible(wk1gt_5,358).
orientation(wk1gt_4,161,359). coords(wk1gt_4,52,176,359).  visible(wk1gt_4,359). orientation(wk1gt_5,162,359). coords(wk1gt_5,50,163,359).  visible(wk1gt_5,359).
orientation(wk1gt_4,162,360). coords(wk1gt_4,52,176,360).  visible(wk1gt_4,360). orientation(wk1gt_5,162,360). coords(wk1gt_5,50,163,360).  visible(wk1gt_5,360).
orientation(wk1gt_4,163,361). coords(wk1gt_4,52,176,361).  visible(wk1gt_4,361). orientation(wk1gt_5,162,361). coords(wk1gt_5,50,163,361).  visible(wk1gt_5,361).
orientation(wk1gt_4,168,362). coords(wk1gt_4,52,176,362).  visible(wk1gt_4,362). orientation(wk1gt_5,162,362). coords(wk1gt_5,50,163,362).  visible(wk1gt_5,362).
orientation(wk1gt_4,164,363). coords(wk1gt_4,52,176,363).  visible(wk1gt_4,363). orientation(wk1gt_5,162,363). coords(wk1gt_5,50,163,363).  visible(wk1gt_5,363).
orientation(wk1gt_4,164,364). coords(wk1gt_4,52,176,364).  visible(wk1gt_4,364). orientation(wk1gt_5,162,364). coords(wk1gt_5,50,163,364).  visible(wk1gt_5,364).
happensAt(disappear(wk1gt_4),363). happensAt(disappear(wk1gt_5),363).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=339..363.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=339..363.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),339). :- holdsAt(meeting(wk1gt_5,wk1gt_4),339).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),340). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),340).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),341). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),341).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),342). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),342).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),343). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),343).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),344). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),344).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),345). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),345).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),346). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),346).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),347). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),347).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),348). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),348).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),349). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),349).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),350). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),350).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),351). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),351).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),352). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),352).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),353). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),353).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),354). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),354).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),355). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),355).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),356). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),356).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),357). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),357).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),358). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),358).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),359). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),359).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),360). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),360).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),361). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),361).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),362). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),362).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),363). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),363).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),364). :- holdsAt(meeting(wk1gt_5,wk1gt_4),364).
:- simpleEvent(T,P,walking), T>339 , T<363.
:- simpleEvent(T,P,running), T>339 , T<363.

#show simpleEvent/3.