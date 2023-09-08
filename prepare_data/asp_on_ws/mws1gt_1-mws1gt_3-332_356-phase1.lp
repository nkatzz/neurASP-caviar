
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

	orientation(mws1gt_1,155,332). coords(mws1gt_1,71,99,332).  visible(mws1gt_1,332). happensAt(walking(mws1gt_1),332). orientation(mws1gt_3,165,332). coords(mws1gt_3,60,146,332).  visible(mws1gt_3,332). happensAt(walking(mws1gt_3),332).
orientation(mws1gt_1,155,333). coords(mws1gt_1,71,99,333).  visible(mws1gt_1,333). happensAt(walking(mws1gt_1),333). orientation(mws1gt_3,165,333). coords(mws1gt_3,60,147,333).  visible(mws1gt_3,333). happensAt(walking(mws1gt_3),333).
orientation(mws1gt_1,155,334). coords(mws1gt_1,71,99,334).  visible(mws1gt_1,334). happensAt(walking(mws1gt_1),334). orientation(mws1gt_3,165,334). coords(mws1gt_3,60,147,334).  visible(mws1gt_3,334). happensAt(walking(mws1gt_3),334).
orientation(mws1gt_1,155,335). coords(mws1gt_1,71,99,335).  visible(mws1gt_1,335). happensAt(walking(mws1gt_1),335). orientation(mws1gt_3,165,335). coords(mws1gt_3,60,148,335).  visible(mws1gt_3,335). happensAt(walking(mws1gt_3),335).
orientation(mws1gt_1,155,336). coords(mws1gt_1,71,98,336).  visible(mws1gt_1,336). happensAt(walking(mws1gt_1),336). orientation(mws1gt_3,165,336). coords(mws1gt_3,60,149,336).  visible(mws1gt_3,336). happensAt(walking(mws1gt_3),336).
orientation(mws1gt_1,155,337). coords(mws1gt_1,71,98,337).  visible(mws1gt_1,337). happensAt(walking(mws1gt_1),337). orientation(mws1gt_3,165,337). coords(mws1gt_3,59,149,337).  visible(mws1gt_3,337). happensAt(walking(mws1gt_3),337).
orientation(mws1gt_1,155,338). coords(mws1gt_1,71,98,338).  visible(mws1gt_1,338). happensAt(walking(mws1gt_1),338). orientation(mws1gt_3,165,338). coords(mws1gt_3,59,149,338).  visible(mws1gt_3,338). happensAt(walking(mws1gt_3),338).
orientation(mws1gt_1,155,339). coords(mws1gt_1,71,97,339).  visible(mws1gt_1,339). happensAt(walking(mws1gt_1),339). orientation(mws1gt_3,165,339). coords(mws1gt_3,59,149,339).  visible(mws1gt_3,339). happensAt(walking(mws1gt_3),339).
orientation(mws1gt_1,143,340). coords(mws1gt_1,70,97,340).  visible(mws1gt_1,340). happensAt(walking(mws1gt_1),340). orientation(mws1gt_3,165,340). coords(mws1gt_3,58,150,340).  visible(mws1gt_3,340). happensAt(walking(mws1gt_3),340).
orientation(mws1gt_1,143,341). coords(mws1gt_1,71,96,341).  visible(mws1gt_1,341). happensAt(walking(mws1gt_1),341). orientation(mws1gt_3,165,341). coords(mws1gt_3,57,150,341).  visible(mws1gt_3,341). happensAt(walking(mws1gt_3),341).
orientation(mws1gt_1,143,342). coords(mws1gt_1,71,96,342).  visible(mws1gt_1,342). happensAt(walking(mws1gt_1),342). orientation(mws1gt_3,165,342). coords(mws1gt_3,56,150,342).  visible(mws1gt_3,342). happensAt(walking(mws1gt_3),342).
orientation(mws1gt_1,143,343). coords(mws1gt_1,71,95,343).  visible(mws1gt_1,343). happensAt(walking(mws1gt_1),343). orientation(mws1gt_3,165,343). coords(mws1gt_3,55,151,343).  visible(mws1gt_3,343). happensAt(walking(mws1gt_3),343).
orientation(mws1gt_1,143,344). coords(mws1gt_1,71,95,344).  visible(mws1gt_1,344). happensAt(walking(mws1gt_1),344). orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344).
orientation(mws1gt_1,143,345). coords(mws1gt_1,70,94,345).  visible(mws1gt_1,345). happensAt(walking(mws1gt_1),345). orientation(mws1gt_3,165,345). coords(mws1gt_3,55,151,345).  visible(mws1gt_3,345). happensAt(walking(mws1gt_3),345).
orientation(mws1gt_1,143,346). coords(mws1gt_1,70,94,346).  visible(mws1gt_1,346). happensAt(walking(mws1gt_1),346). orientation(mws1gt_3,165,346). coords(mws1gt_3,54,152,346).  visible(mws1gt_3,346). happensAt(walking(mws1gt_3),346).
orientation(mws1gt_1,143,347). coords(mws1gt_1,70,93,347).  visible(mws1gt_1,347). happensAt(walking(mws1gt_1),347). orientation(mws1gt_3,165,347). coords(mws1gt_3,54,152,347).  visible(mws1gt_3,347). happensAt(walking(mws1gt_3),347).
orientation(mws1gt_1,143,348). coords(mws1gt_1,70,93,348).  visible(mws1gt_1,348). happensAt(walking(mws1gt_1),348). orientation(mws1gt_3,165,348). coords(mws1gt_3,54,152,348).  visible(mws1gt_3,348). happensAt(walking(mws1gt_3),348).
orientation(mws1gt_1,143,349). coords(mws1gt_1,70,93,349).  visible(mws1gt_1,349). happensAt(walking(mws1gt_1),349). orientation(mws1gt_3,165,349). coords(mws1gt_3,54,153,349).  visible(mws1gt_3,349). happensAt(walking(mws1gt_3),349).
orientation(mws1gt_1,143,350). coords(mws1gt_1,70,92,350).  visible(mws1gt_1,350). happensAt(walking(mws1gt_1),350). orientation(mws1gt_3,165,350). coords(mws1gt_3,54,154,350).  visible(mws1gt_3,350). happensAt(walking(mws1gt_3),350).
orientation(mws1gt_1,143,351). coords(mws1gt_1,70,92,351).  visible(mws1gt_1,351). happensAt(walking(mws1gt_1),351). orientation(mws1gt_3,165,351). coords(mws1gt_3,53,154,351).  visible(mws1gt_3,351). happensAt(walking(mws1gt_3),351).
orientation(mws1gt_1,143,352). coords(mws1gt_1,70,92,352).  visible(mws1gt_1,352). happensAt(walking(mws1gt_1),352). orientation(mws1gt_3,165,352). coords(mws1gt_3,53,154,352).  visible(mws1gt_3,352). happensAt(walking(mws1gt_3),352).
orientation(mws1gt_1,143,353). coords(mws1gt_1,70,92,353).  visible(mws1gt_1,353). happensAt(walking(mws1gt_1),353). orientation(mws1gt_3,165,353). coords(mws1gt_3,53,154,353).  visible(mws1gt_3,353). happensAt(walking(mws1gt_3),353).
orientation(mws1gt_1,143,354). coords(mws1gt_1,70,92,354).  visible(mws1gt_1,354). happensAt(walking(mws1gt_1),354). orientation(mws1gt_3,165,354). coords(mws1gt_3,52,154,354).  visible(mws1gt_3,354). happensAt(walking(mws1gt_3),354).
orientation(mws1gt_1,143,355). coords(mws1gt_1,70,91,355).  visible(mws1gt_1,355). happensAt(walking(mws1gt_1),355). orientation(mws1gt_3,165,355). coords(mws1gt_3,51,154,355).  visible(mws1gt_3,355). happensAt(walking(mws1gt_3),355).
orientation(mws1gt_1,143,356). coords(mws1gt_1,69,90,356).  visible(mws1gt_1,356). happensAt(walking(mws1gt_1),356). orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). happensAt(walking(mws1gt_3),356).
#show holdsAt/2.