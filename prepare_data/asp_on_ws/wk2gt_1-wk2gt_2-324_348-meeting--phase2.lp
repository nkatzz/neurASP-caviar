

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
orientation(wk2gt_1,125,324). coords(wk2gt_1,115,53,324).  visible(wk2gt_1,324). orientation(wk2gt_2,141,324). coords(wk2gt_2,98,63,324).  visible(wk2gt_2,324).
orientation(wk2gt_1,125,325). coords(wk2gt_1,115,53,325).  visible(wk2gt_1,325). orientation(wk2gt_2,141,325). coords(wk2gt_2,98,63,325).  visible(wk2gt_2,325).
orientation(wk2gt_1,125,326). coords(wk2gt_1,115,53,326).  visible(wk2gt_1,326). orientation(wk2gt_2,141,326). coords(wk2gt_2,98,63,326).  visible(wk2gt_2,326).
orientation(wk2gt_1,125,327). coords(wk2gt_1,115,53,327).  visible(wk2gt_1,327). orientation(wk2gt_2,141,327). coords(wk2gt_2,98,63,327).  visible(wk2gt_2,327).
orientation(wk2gt_1,125,328). coords(wk2gt_1,115,53,328).  visible(wk2gt_1,328). orientation(wk2gt_2,141,328). coords(wk2gt_2,98,63,328).  visible(wk2gt_2,328).
orientation(wk2gt_1,125,329). coords(wk2gt_1,115,53,329).  visible(wk2gt_1,329). orientation(wk2gt_2,141,329). coords(wk2gt_2,98,63,329).  visible(wk2gt_2,329).
orientation(wk2gt_1,125,330). coords(wk2gt_1,115,53,330).  visible(wk2gt_1,330). orientation(wk2gt_2,141,330). coords(wk2gt_2,98,63,330).  visible(wk2gt_2,330).
orientation(wk2gt_1,125,331). coords(wk2gt_1,115,53,331).  visible(wk2gt_1,331). orientation(wk2gt_2,141,331). coords(wk2gt_2,98,63,331).  visible(wk2gt_2,331).
orientation(wk2gt_1,125,332). coords(wk2gt_1,115,53,332).  visible(wk2gt_1,332). orientation(wk2gt_2,141,332). coords(wk2gt_2,98,63,332).  visible(wk2gt_2,332).
orientation(wk2gt_1,125,333). coords(wk2gt_1,115,53,333).  visible(wk2gt_1,333). orientation(wk2gt_2,141,333). coords(wk2gt_2,98,63,333).  visible(wk2gt_2,333).
orientation(wk2gt_1,125,334). coords(wk2gt_1,115,53,334).  visible(wk2gt_1,334). orientation(wk2gt_2,141,334). coords(wk2gt_2,98,63,334).  visible(wk2gt_2,334).
orientation(wk2gt_1,125,335). coords(wk2gt_1,115,53,335).  visible(wk2gt_1,335). orientation(wk2gt_2,141,335). coords(wk2gt_2,98,63,335).  visible(wk2gt_2,335).
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
happensAt(disappear(wk2gt_1),348). happensAt(disappear(wk2gt_2),348).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=324..348.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=324..348.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),324). :- holdsAt(meeting(wk2gt_2,wk2gt_1),324).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),325). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),325).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),326). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),326).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),327). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),327).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),328). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),328).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),329). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),329).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),330). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),330).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),331). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),331).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),332). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),332).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),333). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),333).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),334). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),334).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),335). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),335).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),336). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),336).
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
:- holdsAt(meeting(wk2gt_1,wk2gt_2),349). :- holdsAt(meeting(wk2gt_2,wk2gt_1),349).
:- simpleEvent(T,P,walking), T>324 , T<348.
:- simpleEvent(T,P,running), T>324 , T<348.

#show simpleEvent/3.