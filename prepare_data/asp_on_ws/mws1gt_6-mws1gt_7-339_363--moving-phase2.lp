

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

	person(mws1gt_6;mws1gt_7).
orientation(mws1gt_6,146,339). coords(mws1gt_6,32,186,339).  visible(mws1gt_6,339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339).
orientation(mws1gt_6,160,340). coords(mws1gt_6,32,187,340).  visible(mws1gt_6,340). orientation(mws1gt_7,167,340). coords(mws1gt_7,41,193,340).  visible(mws1gt_7,340).
orientation(mws1gt_6,160,341). coords(mws1gt_6,32,187,341).  visible(mws1gt_6,341). orientation(mws1gt_7,167,341). coords(mws1gt_7,41,192,341).  visible(mws1gt_7,341).
orientation(mws1gt_6,160,342). coords(mws1gt_6,32,188,342).  visible(mws1gt_6,342). orientation(mws1gt_7,174,342). coords(mws1gt_7,41,191,342).  visible(mws1gt_7,342).
orientation(mws1gt_6,160,343). coords(mws1gt_6,32,189,343).  visible(mws1gt_6,343). orientation(mws1gt_7,174,343). coords(mws1gt_7,42,191,343).  visible(mws1gt_7,343).
orientation(mws1gt_6,160,344). coords(mws1gt_6,32,189,344).  visible(mws1gt_6,344). orientation(mws1gt_7,174,344). coords(mws1gt_7,44,190,344).  visible(mws1gt_7,344).
orientation(mws1gt_6,160,345). coords(mws1gt_6,32,189,345).  visible(mws1gt_6,345). orientation(mws1gt_7,174,345). coords(mws1gt_7,45,190,345).  visible(mws1gt_7,345).
orientation(mws1gt_6,160,346). coords(mws1gt_6,32,190,346).  visible(mws1gt_6,346). orientation(mws1gt_7,174,346). coords(mws1gt_7,45,190,346).  visible(mws1gt_7,346).
orientation(mws1gt_6,160,347). coords(mws1gt_6,32,191,347).  visible(mws1gt_6,347). orientation(mws1gt_7,174,347). coords(mws1gt_7,44,190,347).  visible(mws1gt_7,347).
orientation(mws1gt_6,160,348). coords(mws1gt_6,32,192,348).  visible(mws1gt_6,348). orientation(mws1gt_7,174,348). coords(mws1gt_7,45,190,348).  visible(mws1gt_7,348).
orientation(mws1gt_6,160,349). coords(mws1gt_6,32,193,349).  visible(mws1gt_6,349). orientation(mws1gt_7,174,349). coords(mws1gt_7,46,189,349).  visible(mws1gt_7,349).
orientation(mws1gt_6,160,350). coords(mws1gt_6,32,194,350).  visible(mws1gt_6,350). orientation(mws1gt_7,174,350). coords(mws1gt_7,46,188,350).  visible(mws1gt_7,350).
orientation(mws1gt_6,160,351). coords(mws1gt_6,32,196,351).  visible(mws1gt_6,351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351).
orientation(mws1gt_6,160,352). coords(mws1gt_6,32,198,352).  visible(mws1gt_6,352). orientation(mws1gt_7,174,352). coords(mws1gt_7,46,188,352).  visible(mws1gt_7,352).
orientation(mws1gt_6,160,353). coords(mws1gt_6,32,199,353).  visible(mws1gt_6,353). orientation(mws1gt_7,174,353). coords(mws1gt_7,46,188,353).  visible(mws1gt_7,353).
orientation(mws1gt_6,160,354). coords(mws1gt_6,32,200,354).  visible(mws1gt_6,354). orientation(mws1gt_7,174,354). coords(mws1gt_7,46,188,354).  visible(mws1gt_7,354).
orientation(mws1gt_6,160,355). coords(mws1gt_6,32,201,355).  visible(mws1gt_6,355). orientation(mws1gt_7,174,355). coords(mws1gt_7,46,187,355).  visible(mws1gt_7,355).
orientation(mws1gt_6,160,356). coords(mws1gt_6,32,201,356).  visible(mws1gt_6,356). orientation(mws1gt_7,174,356). coords(mws1gt_7,46,187,356).  visible(mws1gt_7,356).
orientation(mws1gt_6,160,357). coords(mws1gt_6,32,201,357).  visible(mws1gt_6,357). orientation(mws1gt_7,174,357). coords(mws1gt_7,47,187,357).  visible(mws1gt_7,357).
orientation(mws1gt_6,160,358). coords(mws1gt_6,32,201,358).  visible(mws1gt_6,358). orientation(mws1gt_7,174,358). coords(mws1gt_7,47,187,358).  visible(mws1gt_7,358).
orientation(mws1gt_6,160,359). coords(mws1gt_6,32,202,359).  visible(mws1gt_6,359). orientation(mws1gt_7,174,359). coords(mws1gt_7,49,186,359).  visible(mws1gt_7,359).
orientation(mws1gt_6,160,360). coords(mws1gt_6,32,203,360).  visible(mws1gt_6,360). orientation(mws1gt_7,174,360). coords(mws1gt_7,50,185,360).  visible(mws1gt_7,360).
orientation(mws1gt_6,160,361). coords(mws1gt_6,32,203,361).  visible(mws1gt_6,361). orientation(mws1gt_7,174,361). coords(mws1gt_7,50,185,361).  visible(mws1gt_7,361).
orientation(mws1gt_6,160,362). coords(mws1gt_6,32,204,362).  visible(mws1gt_6,362). orientation(mws1gt_7,174,362). coords(mws1gt_7,50,185,362).  visible(mws1gt_7,362).
orientation(mws1gt_6,160,363). coords(mws1gt_6,31,204,363).  visible(mws1gt_6,363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363).
orientation(mws1gt_6,160,364). coords(mws1gt_6,31,204,364).  visible(mws1gt_6,364). orientation(mws1gt_7,174,364). coords(mws1gt_7,50,185,364).  visible(mws1gt_7,364).
happensAt(disappear(mws1gt_6),363). happensAt(disappear(mws1gt_7),363).
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=339..363.
1{simpleEvent(T,mws1gt_7,active); simpleEvent(T,mws1gt_7,inactive); simpleEvent(T,mws1gt_7,walking); simpleEvent(T,mws1gt_7,running)}1 :- T=339..363.
:- holdsAt(moving(mws1gt_6,mws1gt_7),339). :- holdsAt(moving(mws1gt_7,mws1gt_6),339).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),340). :- not holdsAt(moving(mws1gt_7,mws1gt_6),340).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),341). :- not holdsAt(moving(mws1gt_7,mws1gt_6),341).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),342). :- not holdsAt(moving(mws1gt_7,mws1gt_6),342).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),343). :- not holdsAt(moving(mws1gt_7,mws1gt_6),343).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),344). :- not holdsAt(moving(mws1gt_7,mws1gt_6),344).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),345). :- not holdsAt(moving(mws1gt_7,mws1gt_6),345).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),346). :- not holdsAt(moving(mws1gt_7,mws1gt_6),346).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),347). :- not holdsAt(moving(mws1gt_7,mws1gt_6),347).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),348). :- not holdsAt(moving(mws1gt_7,mws1gt_6),348).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),349). :- not holdsAt(moving(mws1gt_7,mws1gt_6),349).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),350). :- not holdsAt(moving(mws1gt_7,mws1gt_6),350).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),351). :- not holdsAt(moving(mws1gt_7,mws1gt_6),351).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),352). :- not holdsAt(moving(mws1gt_7,mws1gt_6),352).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),353). :- not holdsAt(moving(mws1gt_7,mws1gt_6),353).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),354). :- not holdsAt(moving(mws1gt_7,mws1gt_6),354).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),355). :- not holdsAt(moving(mws1gt_7,mws1gt_6),355).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),356). :- not holdsAt(moving(mws1gt_7,mws1gt_6),356).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),357). :- not holdsAt(moving(mws1gt_7,mws1gt_6),357).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),358). :- not holdsAt(moving(mws1gt_7,mws1gt_6),358).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),359). :- not holdsAt(moving(mws1gt_7,mws1gt_6),359).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),360). :- not holdsAt(moving(mws1gt_7,mws1gt_6),360).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),361). :- not holdsAt(moving(mws1gt_7,mws1gt_6),361).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),362). :- not holdsAt(moving(mws1gt_7,mws1gt_6),362).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),363). :- not holdsAt(moving(mws1gt_7,mws1gt_6),363).
:- holdsAt(moving(mws1gt_6,mws1gt_7),364). :- holdsAt(moving(mws1gt_7,mws1gt_6),364).
:- simpleEvent(T,P,inactive), T>339 , T<363.
:- simpleEvent(T,P,running), T>339 , T<363.

#show simpleEvent/3.