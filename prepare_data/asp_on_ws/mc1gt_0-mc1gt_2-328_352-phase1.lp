
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(mc1gt_0,16,328). coords(mc1gt_0,214,269,328).  visible(mc1gt_0,328). happensAt(walking(mc1gt_0),328). orientation(mc1gt_2,13,328). coords(mc1gt_2,236,240,328).  visible(mc1gt_2,328). happensAt(walking(mc1gt_2),328).
orientation(mc1gt_0,19,329). coords(mc1gt_0,214,269,329).  visible(mc1gt_0,329). happensAt(walking(mc1gt_0),329). orientation(mc1gt_2,13,329). coords(mc1gt_2,238,241,329).  visible(mc1gt_2,329). happensAt(walking(mc1gt_2),329).
orientation(mc1gt_0,22,330). coords(mc1gt_0,214,270,330).  visible(mc1gt_0,330). happensAt(walking(mc1gt_0),330). orientation(mc1gt_2,13,330). coords(mc1gt_2,240,243,330).  visible(mc1gt_2,330). happensAt(walking(mc1gt_2),330).
orientation(mc1gt_0,22,331). coords(mc1gt_0,214,272,331).  visible(mc1gt_0,331). happensAt(walking(mc1gt_0),331). orientation(mc1gt_2,13,331). coords(mc1gt_2,242,244,331).  visible(mc1gt_2,331). happensAt(walking(mc1gt_2),331).
orientation(mc1gt_0,22,332). coords(mc1gt_0,214,272,332).  visible(mc1gt_0,332). happensAt(walking(mc1gt_0),332). orientation(mc1gt_2,13,332). coords(mc1gt_2,243,245,332).  visible(mc1gt_2,332). happensAt(walking(mc1gt_2),332).
orientation(mc1gt_0,22,333). coords(mc1gt_0,214,272,333).  visible(mc1gt_0,333). happensAt(walking(mc1gt_0),333). orientation(mc1gt_2,13,333). coords(mc1gt_2,244,245,333).  visible(mc1gt_2,333). happensAt(walking(mc1gt_2),333).
orientation(mc1gt_0,22,334). coords(mc1gt_0,214,272,334).  visible(mc1gt_0,334). happensAt(walking(mc1gt_0),334). orientation(mc1gt_2,13,334). coords(mc1gt_2,244,245,334).  visible(mc1gt_2,334). happensAt(walking(mc1gt_2),334).
orientation(mc1gt_0,22,335). coords(mc1gt_0,214,272,335).  visible(mc1gt_0,335). happensAt(walking(mc1gt_0),335). orientation(mc1gt_2,13,335). coords(mc1gt_2,245,246,335).  visible(mc1gt_2,335). happensAt(walking(mc1gt_2),335).
orientation(mc1gt_0,22,336). coords(mc1gt_0,214,273,336).  visible(mc1gt_0,336). happensAt(walking(mc1gt_0),336). orientation(mc1gt_2,13,336). coords(mc1gt_2,245,247,336).  visible(mc1gt_2,336). happensAt(walking(mc1gt_2),336).
orientation(mc1gt_0,22,337). coords(mc1gt_0,214,274,337).  visible(mc1gt_0,337). happensAt(walking(mc1gt_0),337). orientation(mc1gt_2,13,337). coords(mc1gt_2,246,248,337).  visible(mc1gt_2,337). happensAt(walking(mc1gt_2),337).
orientation(mc1gt_0,22,338). coords(mc1gt_0,216,275,338).  visible(mc1gt_0,338). happensAt(walking(mc1gt_0),338). orientation(mc1gt_2,13,338). coords(mc1gt_2,247,250,338).  visible(mc1gt_2,338). happensAt(walking(mc1gt_2),338).
orientation(mc1gt_0,22,339). coords(mc1gt_0,216,276,339).  visible(mc1gt_0,339). happensAt(walking(mc1gt_0),339). orientation(mc1gt_2,13,339). coords(mc1gt_2,247,253,339).  visible(mc1gt_2,339). happensAt(walking(mc1gt_2),339).
orientation(mc1gt_0,22,340). coords(mc1gt_0,217,278,340).  visible(mc1gt_0,340). happensAt(walking(mc1gt_0),340). orientation(mc1gt_2,13,340). coords(mc1gt_2,247,254,340).  visible(mc1gt_2,340). happensAt(walking(mc1gt_2),340).
orientation(mc1gt_0,22,341). coords(mc1gt_0,218,280,341).  visible(mc1gt_0,341). happensAt(walking(mc1gt_0),341). orientation(mc1gt_2,13,341). coords(mc1gt_2,247,258,341).  visible(mc1gt_2,341). happensAt(walking(mc1gt_2),341).
orientation(mc1gt_0,22,342). coords(mc1gt_0,219,281,342).  visible(mc1gt_0,342). happensAt(walking(mc1gt_0),342). orientation(mc1gt_2,13,342). coords(mc1gt_2,247,258,342).  visible(mc1gt_2,342). happensAt(walking(mc1gt_2),342).
orientation(mc1gt_0,22,343). coords(mc1gt_0,221,281,343).  visible(mc1gt_0,343). happensAt(walking(mc1gt_0),343). orientation(mc1gt_2,13,343). coords(mc1gt_2,248,259,343).  visible(mc1gt_2,343). happensAt(walking(mc1gt_2),343).
orientation(mc1gt_0,22,344). coords(mc1gt_0,221,281,344).  visible(mc1gt_0,344). happensAt(walking(mc1gt_0),344). orientation(mc1gt_2,16,344). coords(mc1gt_2,248,260,344).  visible(mc1gt_2,344). happensAt(walking(mc1gt_2),344).
orientation(mc1gt_0,28,345). coords(mc1gt_0,220,281,345).  visible(mc1gt_0,345). happensAt(walking(mc1gt_0),345). orientation(mc1gt_2,15,345). coords(mc1gt_2,249,262,345).  visible(mc1gt_2,345). happensAt(walking(mc1gt_2),345).
orientation(mc1gt_0,31,346). coords(mc1gt_0,221,281,346).  visible(mc1gt_0,346). happensAt(walking(mc1gt_0),346). orientation(mc1gt_2,13,346). coords(mc1gt_2,250,263,346).  visible(mc1gt_2,346). happensAt(walking(mc1gt_2),346).
orientation(mc1gt_0,33,347). coords(mc1gt_0,221,281,347).  visible(mc1gt_0,347). happensAt(walking(mc1gt_0),347). orientation(mc1gt_2,16,347). coords(mc1gt_2,250,263,347).  visible(mc1gt_2,347). happensAt(walking(mc1gt_2),347).
orientation(mc1gt_0,39,348). coords(mc1gt_0,223,281,348).  visible(mc1gt_0,348). happensAt(walking(mc1gt_0),348). orientation(mc1gt_2,17,348). coords(mc1gt_2,250,264,348).  visible(mc1gt_2,348). happensAt(walking(mc1gt_2),348).
orientation(mc1gt_0,53,349). coords(mc1gt_0,226,281,349).  visible(mc1gt_0,349). happensAt(walking(mc1gt_0),349). orientation(mc1gt_2,26,349). coords(mc1gt_2,250,265,349).  visible(mc1gt_2,349). happensAt(walking(mc1gt_2),349).
orientation(mc1gt_0,61,350). coords(mc1gt_0,227,281,350).  visible(mc1gt_0,350). happensAt(walking(mc1gt_0),350). orientation(mc1gt_2,35,350). coords(mc1gt_2,250,266,350).  visible(mc1gt_2,350). happensAt(walking(mc1gt_2),350).
orientation(mc1gt_0,70,351). coords(mc1gt_0,227,281,351).  visible(mc1gt_0,351). happensAt(walking(mc1gt_0),351). orientation(mc1gt_2,43,351). coords(mc1gt_2,250,267,351).  visible(mc1gt_2,351). happensAt(walking(mc1gt_2),351).
orientation(mc1gt_0,90,352). coords(mc1gt_0,227,284,352).  happensAt(disappear(mc1gt_0),352). happensAt(walking(mc1gt_0),352). orientation(mc1gt_2,43,352). coords(mc1gt_2,250,268,352).  visible(mc1gt_2,352). happensAt(walking(mc1gt_2),352).
#show holdsAt/2.