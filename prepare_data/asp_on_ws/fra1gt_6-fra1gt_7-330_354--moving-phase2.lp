

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

	person(fra1gt_6;fra1gt_7).
orientation(fra1gt_6,134,330). coords(fra1gt_6,155,76,330).  visible(fra1gt_6,330). orientation(fra1gt_7,142,330). coords(fra1gt_7,157,65,330).  visible(fra1gt_7,330).
orientation(fra1gt_6,134,331). coords(fra1gt_6,155,75,331).  visible(fra1gt_6,331). orientation(fra1gt_7,142,331). coords(fra1gt_7,158,65,331).  visible(fra1gt_7,331).
orientation(fra1gt_6,134,332). coords(fra1gt_6,156,74,332).  visible(fra1gt_6,332). orientation(fra1gt_7,142,332). coords(fra1gt_7,160,65,332).  visible(fra1gt_7,332).
orientation(fra1gt_6,134,333). coords(fra1gt_6,157,74,333).  visible(fra1gt_6,333). orientation(fra1gt_7,142,333). coords(fra1gt_7,159,67,333).  visible(fra1gt_7,333).
orientation(fra1gt_6,134,334). coords(fra1gt_6,158,76,334).  visible(fra1gt_6,334). orientation(fra1gt_7,142,334). coords(fra1gt_7,159,68,334).  visible(fra1gt_7,334).
orientation(fra1gt_6,134,335). coords(fra1gt_6,157,76,335).  visible(fra1gt_6,335). orientation(fra1gt_7,142,335). coords(fra1gt_7,161,68,335).  visible(fra1gt_7,335).
orientation(fra1gt_6,134,336). coords(fra1gt_6,157,76,336).  visible(fra1gt_6,336). orientation(fra1gt_7,142,336). coords(fra1gt_7,163,69,336).  visible(fra1gt_7,336).
orientation(fra1gt_6,134,337). coords(fra1gt_6,157,76,337).  visible(fra1gt_6,337). orientation(fra1gt_7,142,337). coords(fra1gt_7,165,69,337).  visible(fra1gt_7,337).
orientation(fra1gt_6,134,338). coords(fra1gt_6,157,77,338).  visible(fra1gt_6,338). orientation(fra1gt_7,142,338). coords(fra1gt_7,165,69,338).  visible(fra1gt_7,338).
orientation(fra1gt_6,134,339). coords(fra1gt_6,158,77,339).  visible(fra1gt_6,339). orientation(fra1gt_7,142,339). coords(fra1gt_7,165,69,339).  visible(fra1gt_7,339).
orientation(fra1gt_6,134,340). coords(fra1gt_6,158,77,340).  visible(fra1gt_6,340). orientation(fra1gt_7,142,340). coords(fra1gt_7,167,70,340).  visible(fra1gt_7,340).
orientation(fra1gt_6,134,341). coords(fra1gt_6,158,77,341).  visible(fra1gt_6,341). orientation(fra1gt_7,142,341). coords(fra1gt_7,168,70,341).  visible(fra1gt_7,341).
orientation(fra1gt_6,134,342). coords(fra1gt_6,160,77,342).  visible(fra1gt_6,342). orientation(fra1gt_7,142,342). coords(fra1gt_7,167,70,342).  visible(fra1gt_7,342).
orientation(fra1gt_6,134,343). coords(fra1gt_6,160,78,343).  visible(fra1gt_6,343). orientation(fra1gt_7,142,343). coords(fra1gt_7,168,70,343).  visible(fra1gt_7,343).
orientation(fra1gt_6,134,344). coords(fra1gt_6,161,78,344).  visible(fra1gt_6,344). orientation(fra1gt_7,142,344). coords(fra1gt_7,169,70,344).  visible(fra1gt_7,344).
orientation(fra1gt_6,134,345). coords(fra1gt_6,161,78,345).  visible(fra1gt_6,345). orientation(fra1gt_7,142,345). coords(fra1gt_7,169,69,345).  visible(fra1gt_7,345).
orientation(fra1gt_6,134,346). coords(fra1gt_6,161,78,346).  visible(fra1gt_6,346). orientation(fra1gt_7,142,346). coords(fra1gt_7,169,69,346).  visible(fra1gt_7,346).
orientation(fra1gt_6,134,347). coords(fra1gt_6,162,78,347).  visible(fra1gt_6,347). orientation(fra1gt_7,150,347). coords(fra1gt_7,169,68,347).  visible(fra1gt_7,347).
orientation(fra1gt_6,134,348). coords(fra1gt_6,162,78,348).  visible(fra1gt_6,348). orientation(fra1gt_7,150,348). coords(fra1gt_7,169,68,348).  visible(fra1gt_7,348).
orientation(fra1gt_6,134,349). coords(fra1gt_6,162,78,349).  visible(fra1gt_6,349). orientation(fra1gt_7,150,349). coords(fra1gt_7,171,68,349).  visible(fra1gt_7,349).
orientation(fra1gt_6,134,350). coords(fra1gt_6,162,80,350).  visible(fra1gt_6,350). orientation(fra1gt_7,155,350). coords(fra1gt_7,171,68,350).  visible(fra1gt_7,350).
orientation(fra1gt_6,134,351). coords(fra1gt_6,161,82,351).  visible(fra1gt_6,351). orientation(fra1gt_7,155,351). coords(fra1gt_7,173,68,351).  visible(fra1gt_7,351).
orientation(fra1gt_6,134,352). coords(fra1gt_6,159,84,352).  visible(fra1gt_6,352). orientation(fra1gt_7,155,352). coords(fra1gt_7,174,68,352).  visible(fra1gt_7,352).
orientation(fra1gt_6,134,353). coords(fra1gt_6,159,84,353).  visible(fra1gt_6,353). orientation(fra1gt_7,155,353). coords(fra1gt_7,175,68,353).  visible(fra1gt_7,353).
orientation(fra1gt_6,134,354). coords(fra1gt_6,159,86,354).  visible(fra1gt_6,354). orientation(fra1gt_7,155,354). coords(fra1gt_7,176,68,354).  visible(fra1gt_7,354).
orientation(fra1gt_6,134,355). coords(fra1gt_6,159,86,355).  visible(fra1gt_6,355). orientation(fra1gt_7,155,355). coords(fra1gt_7,176,68,355).  visible(fra1gt_7,355).
happensAt(disappear(fra1gt_6),354). happensAt(disappear(fra1gt_7),354).
1{simpleEvent(T,fra1gt_6,active); simpleEvent(T,fra1gt_6,inactive); simpleEvent(T,fra1gt_6,walking); simpleEvent(T,fra1gt_6,running)}1 :- T=330..354.
1{simpleEvent(T,fra1gt_7,active); simpleEvent(T,fra1gt_7,inactive); simpleEvent(T,fra1gt_7,walking); simpleEvent(T,fra1gt_7,running)}1 :- T=330..354.
:- holdsAt(moving(fra1gt_6,fra1gt_7),330). :- holdsAt(moving(fra1gt_7,fra1gt_6),330).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),331). :- not holdsAt(moving(fra1gt_7,fra1gt_6),331).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),332). :- not holdsAt(moving(fra1gt_7,fra1gt_6),332).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),333). :- not holdsAt(moving(fra1gt_7,fra1gt_6),333).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),334). :- not holdsAt(moving(fra1gt_7,fra1gt_6),334).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),335). :- not holdsAt(moving(fra1gt_7,fra1gt_6),335).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),336). :- not holdsAt(moving(fra1gt_7,fra1gt_6),336).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),337). :- not holdsAt(moving(fra1gt_7,fra1gt_6),337).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),338). :- not holdsAt(moving(fra1gt_7,fra1gt_6),338).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),339). :- not holdsAt(moving(fra1gt_7,fra1gt_6),339).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),340). :- not holdsAt(moving(fra1gt_7,fra1gt_6),340).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),341). :- not holdsAt(moving(fra1gt_7,fra1gt_6),341).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),342). :- not holdsAt(moving(fra1gt_7,fra1gt_6),342).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),343). :- not holdsAt(moving(fra1gt_7,fra1gt_6),343).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),344). :- not holdsAt(moving(fra1gt_7,fra1gt_6),344).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),345). :- not holdsAt(moving(fra1gt_7,fra1gt_6),345).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),346). :- not holdsAt(moving(fra1gt_7,fra1gt_6),346).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),347). :- not holdsAt(moving(fra1gt_7,fra1gt_6),347).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),348). :- not holdsAt(moving(fra1gt_7,fra1gt_6),348).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),349). :- not holdsAt(moving(fra1gt_7,fra1gt_6),349).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),350). :- not holdsAt(moving(fra1gt_7,fra1gt_6),350).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),351). :- not holdsAt(moving(fra1gt_7,fra1gt_6),351).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),352). :- not holdsAt(moving(fra1gt_7,fra1gt_6),352).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),353). :- not holdsAt(moving(fra1gt_7,fra1gt_6),353).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),354). :- not holdsAt(moving(fra1gt_7,fra1gt_6),354).
:- holdsAt(moving(fra1gt_6,fra1gt_7),355). :- holdsAt(moving(fra1gt_7,fra1gt_6),355).
:- simpleEvent(T,P,inactive), T>330 , T<354.
:- simpleEvent(T,P,running), T>330 , T<354.

#show simpleEvent/3.