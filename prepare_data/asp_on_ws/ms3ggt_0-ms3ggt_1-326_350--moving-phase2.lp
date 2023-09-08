

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
orientation(ms3ggt_0,155,326). coords(ms3ggt_0,95,145,326).  visible(ms3ggt_0,326). orientation(ms3ggt_1,162,326). coords(ms3ggt_1,93,167,326).  visible(ms3ggt_1,326).
orientation(ms3ggt_0,155,327). coords(ms3ggt_0,96,145,327).  visible(ms3ggt_0,327). orientation(ms3ggt_1,162,327). coords(ms3ggt_1,93,167,327).  visible(ms3ggt_1,327).
orientation(ms3ggt_0,155,328). coords(ms3ggt_0,96,145,328).  visible(ms3ggt_0,328). orientation(ms3ggt_1,162,328). coords(ms3ggt_1,93,167,328).  visible(ms3ggt_1,328).
orientation(ms3ggt_0,155,329). coords(ms3ggt_0,98,145,329).  visible(ms3ggt_0,329). orientation(ms3ggt_1,156,329). coords(ms3ggt_1,94,167,329).  visible(ms3ggt_1,329).
orientation(ms3ggt_0,159,330). coords(ms3ggt_0,100,145,330).  visible(ms3ggt_0,330). orientation(ms3ggt_1,156,330). coords(ms3ggt_1,95,167,330).  visible(ms3ggt_1,330).
orientation(ms3ggt_0,159,331). coords(ms3ggt_0,101,145,331).  visible(ms3ggt_0,331). orientation(ms3ggt_1,156,331). coords(ms3ggt_1,95,167,331).  visible(ms3ggt_1,331).
orientation(ms3ggt_0,163,332). coords(ms3ggt_0,103,145,332).  visible(ms3ggt_0,332). orientation(ms3ggt_1,156,332). coords(ms3ggt_1,97,167,332).  visible(ms3ggt_1,332).
orientation(ms3ggt_0,163,333). coords(ms3ggt_0,104,145,333).  visible(ms3ggt_0,333). orientation(ms3ggt_1,156,333). coords(ms3ggt_1,98,167,333).  visible(ms3ggt_1,333).
orientation(ms3ggt_0,163,334). coords(ms3ggt_0,105,145,334).  visible(ms3ggt_0,334). orientation(ms3ggt_1,156,334). coords(ms3ggt_1,100,167,334).  visible(ms3ggt_1,334).
orientation(ms3ggt_0,163,335). coords(ms3ggt_0,106,145,335).  visible(ms3ggt_0,335). orientation(ms3ggt_1,156,335). coords(ms3ggt_1,101,166,335).  visible(ms3ggt_1,335).
orientation(ms3ggt_0,163,336). coords(ms3ggt_0,106,145,336).  visible(ms3ggt_0,336). orientation(ms3ggt_1,156,336). coords(ms3ggt_1,103,166,336).  visible(ms3ggt_1,336).
orientation(ms3ggt_0,163,337). coords(ms3ggt_0,106,145,337).  visible(ms3ggt_0,337). orientation(ms3ggt_1,156,337). coords(ms3ggt_1,103,166,337).  visible(ms3ggt_1,337).
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
orientation(ms3ggt_0,158,351). coords(ms3ggt_0,120,145,351).  visible(ms3ggt_0,351). orientation(ms3ggt_1,156,351). coords(ms3ggt_1,114,164,351).  visible(ms3ggt_1,351).
happensAt(disappear(ms3ggt_0),350). happensAt(disappear(ms3ggt_1),350).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=326..350.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=326..350.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),326). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),326).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),327). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),327).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),328). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),328).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),329). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),329).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),330). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),330).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),331). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),331).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),332). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),332).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),333). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),333).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),334). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),334).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),335). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),335).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),336). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),336).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),337). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),337).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),338). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),338).
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
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),351). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),351).
:- simpleEvent(T,P,inactive), T>326 , T<350.
:- simpleEvent(T,P,running), T>326 , T<350.

#show simpleEvent/3.