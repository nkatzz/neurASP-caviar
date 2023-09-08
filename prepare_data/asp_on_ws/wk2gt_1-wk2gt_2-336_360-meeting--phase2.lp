

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
orientation(wk2gt_1,125,336). coords(wk2gt_1,115,53,336).  visible(wk2gt_1,336). orientation(wk2gt_2,141,336). coords(wk2gt_2,98,63,336).  visible(wk2gt_2,336).
orientation(wk2gt_1,125,337). coords(wk2gt_1,115,53,337).  visible(wk2gt_1,337). orientation(wk2gt_2,141,337). coords(wk2gt_2,98,63,337).  visible(wk2gt_2,337).
orientation(wk2gt_1,125,338). coords(wk2gt_1,115,53,338).  visible(wk2gt_1,338). orientation(wk2gt_2,141,338). coords(wk2gt_2,98,63,338).  visible(wk2gt_2,338).
orientation(wk2gt_1,125,339). coords(wk2gt_1,115,53,339).  visible(wk2gt_1,339). orientation(wk2gt_2,141,339). coords(wk2gt_2,98,63,339).  visible(wk2gt_2,339).
orientation(wk2gt_1,125,340). coords(wk2gt_1,115,53,340).  visible(wk2gt_1,340). orientation(wk2gt_2,141,340). coords(wk2gt_2,98,63,340).  visible(wk2gt_2,340).
orientation(wk2gt_1,125,341). coords(wk2gt_1,115,53,341).  visible(wk2gt_1,341). orientation(wk2gt_2,141,341). coords(wk2gt_2,98,63,341).  visible(wk2gt_2,341).
orientation(wk2gt_1,125,342). coords(wk2gt_1,115,53,342).  visible(wk2gt_1,342). orientation(wk2gt_2,141,342). coords(wk2gt_2,98,63,342).  visible(wk2gt_2,342).
orientation(wk2gt_1,125,343). coords(wk2gt_1,115,53,343).  visible(wk2gt_1,343). orientation(wk2gt_2,141,343). coords(wk2gt_2,98,63,343).  visible(wk2gt_2,343).
orientation(wk2gt_1,125,344). coords(wk2gt_1,115,53,344).  visible(wk2gt_1,344). orientation(wk2gt_2,141,344). coords(wk2gt_2,98,63,344).  visible(wk2gt_2,344).
orientation(wk2gt_1,125,345). coords(wk2gt_1,115,53,345).  visible(wk2gt_1,345). orientation(wk2gt_2,141,345). coords(wk2gt_2,98,63,345).  visible(wk2gt_2,345).
orientation(wk2gt_1,125,346). coords(wk2gt_1,115,53,346).  visible(wk2gt_1,346). orientation(wk2gt_2,141,346). coords(wk2gt_2,98,63,346).  visible(wk2gt_2,346).
orientation(wk2gt_1,125,347). coords(wk2gt_1,115,53,347).  visible(wk2gt_1,347). orientation(wk2gt_2,141,347). coords(wk2gt_2,98,63,347).  visible(wk2gt_2,347).
orientation(wk2gt_1,125,348). coords(wk2gt_1,115,53,348).  visible(wk2gt_1,348). orientation(wk2gt_2,141,348). coords(wk2gt_2,98,63,348).  visible(wk2gt_2,348).
orientation(wk2gt_1,125,349). coords(wk2gt_1,115,53,349).  visible(wk2gt_1,349). orientation(wk2gt_2,141,349). coords(wk2gt_2,98,63,349).  visible(wk2gt_2,349).
orientation(wk2gt_1,125,350). coords(wk2gt_1,115,53,350).  visible(wk2gt_1,350). orientation(wk2gt_2,141,350). coords(wk2gt_2,98,63,350).  visible(wk2gt_2,350).
orientation(wk2gt_1,125,351). coords(wk2gt_1,115,53,351).  visible(wk2gt_1,351). orientation(wk2gt_2,141,351). coords(wk2gt_2,98,63,351).  visible(wk2gt_2,351).
orientation(wk2gt_1,125,352). coords(wk2gt_1,115,53,352).  visible(wk2gt_1,352). orientation(wk2gt_2,141,352). coords(wk2gt_2,98,63,352).  visible(wk2gt_2,352).
orientation(wk2gt_1,125,353). coords(wk2gt_1,115,53,353).  visible(wk2gt_1,353). orientation(wk2gt_2,141,353). coords(wk2gt_2,98,63,353).  visible(wk2gt_2,353).
orientation(wk2gt_1,125,354). coords(wk2gt_1,115,53,354).  visible(wk2gt_1,354). orientation(wk2gt_2,141,354). coords(wk2gt_2,98,63,354).  visible(wk2gt_2,354).
orientation(wk2gt_1,125,355). coords(wk2gt_1,115,53,355).  visible(wk2gt_1,355). orientation(wk2gt_2,141,355). coords(wk2gt_2,98,63,355).  visible(wk2gt_2,355).
orientation(wk2gt_1,125,356). coords(wk2gt_1,115,53,356).  visible(wk2gt_1,356). orientation(wk2gt_2,141,356). coords(wk2gt_2,98,63,356).  visible(wk2gt_2,356).
orientation(wk2gt_1,125,357). coords(wk2gt_1,115,53,357).  visible(wk2gt_1,357). orientation(wk2gt_2,141,357). coords(wk2gt_2,98,63,357).  visible(wk2gt_2,357).
orientation(wk2gt_1,125,358). coords(wk2gt_1,115,53,358).  visible(wk2gt_1,358). orientation(wk2gt_2,141,358). coords(wk2gt_2,98,63,358).  visible(wk2gt_2,358).
orientation(wk2gt_1,125,359). coords(wk2gt_1,115,53,359).  visible(wk2gt_1,359). orientation(wk2gt_2,141,359). coords(wk2gt_2,98,63,359).  visible(wk2gt_2,359).
orientation(wk2gt_1,125,360). coords(wk2gt_1,115,53,360).  visible(wk2gt_1,360). orientation(wk2gt_2,141,360). coords(wk2gt_2,98,63,360).  visible(wk2gt_2,360).
orientation(wk2gt_1,125,361). coords(wk2gt_1,115,53,361).  visible(wk2gt_1,361). orientation(wk2gt_2,141,361). coords(wk2gt_2,98,63,361).  visible(wk2gt_2,361).
happensAt(disappear(wk2gt_1),360). happensAt(disappear(wk2gt_2),360).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=336..360.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=336..360.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),336). :- holdsAt(meeting(wk2gt_2,wk2gt_1),336).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),337). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),337).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),338). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),338).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),339). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),339).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),340). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),340).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),341). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),341).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),342). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),342).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),343). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),343).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),344). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),344).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),345). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),345).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),346). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),346).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),347). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),347).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),348). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),348).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),349). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),349).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),350). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),350).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),351). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),351).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),352). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),352).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),353). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),353).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),354). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),354).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),355). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),355).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),356). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),356).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),357). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),357).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),358). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),358).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),359). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),359).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),360). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),360).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),361). :- holdsAt(meeting(wk2gt_2,wk2gt_1),361).
:- simpleEvent(T,P,walking), T>336 , T<360.
:- simpleEvent(T,P,running), T>336 , T<360.

#show simpleEvent/3.