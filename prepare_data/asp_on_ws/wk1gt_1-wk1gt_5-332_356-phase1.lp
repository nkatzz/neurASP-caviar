
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

	orientation(wk1gt_1,148,332). coords(wk1gt_1,156,148,332).  visible(wk1gt_1,332). happensAt(walking(wk1gt_1),332). orientation(wk1gt_5,162,332). coords(wk1gt_5,50,163,332).  visible(wk1gt_5,332). happensAt(inactive(wk1gt_5),332).
orientation(wk1gt_1,148,333). coords(wk1gt_1,154,146,333).  visible(wk1gt_1,333). happensAt(walking(wk1gt_1),333). orientation(wk1gt_5,162,333). coords(wk1gt_5,50,163,333).  visible(wk1gt_5,333). happensAt(inactive(wk1gt_5),333).
orientation(wk1gt_1,148,334). coords(wk1gt_1,153,145,334).  visible(wk1gt_1,334). happensAt(walking(wk1gt_1),334). orientation(wk1gt_5,162,334). coords(wk1gt_5,50,163,334).  visible(wk1gt_5,334). happensAt(inactive(wk1gt_5),334).
orientation(wk1gt_1,148,335). coords(wk1gt_1,153,144,335).  visible(wk1gt_1,335). happensAt(walking(wk1gt_1),335). orientation(wk1gt_5,162,335). coords(wk1gt_5,50,163,335).  visible(wk1gt_5,335). happensAt(inactive(wk1gt_5),335).
orientation(wk1gt_1,153,336). coords(wk1gt_1,152,141,336).  visible(wk1gt_1,336). happensAt(walking(wk1gt_1),336). orientation(wk1gt_5,162,336). coords(wk1gt_5,50,163,336).  visible(wk1gt_5,336). happensAt(inactive(wk1gt_5),336).
orientation(wk1gt_1,153,337). coords(wk1gt_1,152,141,337).  visible(wk1gt_1,337). happensAt(walking(wk1gt_1),337). orientation(wk1gt_5,162,337). coords(wk1gt_5,50,163,337).  visible(wk1gt_5,337). happensAt(inactive(wk1gt_5),337).
orientation(wk1gt_1,153,338). coords(wk1gt_1,151,140,338).  visible(wk1gt_1,338). happensAt(walking(wk1gt_1),338). orientation(wk1gt_5,162,338). coords(wk1gt_5,50,163,338).  visible(wk1gt_5,338). happensAt(inactive(wk1gt_5),338).
orientation(wk1gt_1,153,339). coords(wk1gt_1,151,140,339).  visible(wk1gt_1,339). happensAt(walking(wk1gt_1),339). orientation(wk1gt_5,162,339). coords(wk1gt_5,50,163,339).  visible(wk1gt_5,339). happensAt(inactive(wk1gt_5),339).
orientation(wk1gt_1,149,340). coords(wk1gt_1,150,140,340).  visible(wk1gt_1,340). happensAt(walking(wk1gt_1),340). orientation(wk1gt_5,162,340). coords(wk1gt_5,50,163,340).  visible(wk1gt_5,340). happensAt(inactive(wk1gt_5),340).
orientation(wk1gt_1,149,341). coords(wk1gt_1,150,139,341).  visible(wk1gt_1,341). happensAt(walking(wk1gt_1),341). orientation(wk1gt_5,162,341). coords(wk1gt_5,50,163,341).  visible(wk1gt_5,341). happensAt(inactive(wk1gt_5),341).
orientation(wk1gt_1,149,342). coords(wk1gt_1,149,139,342).  visible(wk1gt_1,342). happensAt(walking(wk1gt_1),342). orientation(wk1gt_5,162,342). coords(wk1gt_5,50,163,342).  visible(wk1gt_5,342). happensAt(inactive(wk1gt_5),342).
orientation(wk1gt_1,149,343). coords(wk1gt_1,148,138,343).  visible(wk1gt_1,343). happensAt(walking(wk1gt_1),343). orientation(wk1gt_5,162,343). coords(wk1gt_5,50,163,343).  visible(wk1gt_5,343). happensAt(inactive(wk1gt_5),343).
orientation(wk1gt_1,149,344). coords(wk1gt_1,147,137,344).  visible(wk1gt_1,344). happensAt(walking(wk1gt_1),344). orientation(wk1gt_5,162,344). coords(wk1gt_5,50,163,344).  visible(wk1gt_5,344). happensAt(inactive(wk1gt_5),344).
orientation(wk1gt_1,149,345). coords(wk1gt_1,145,136,345).  visible(wk1gt_1,345). happensAt(walking(wk1gt_1),345). orientation(wk1gt_5,162,345). coords(wk1gt_5,50,163,345).  visible(wk1gt_5,345). happensAt(inactive(wk1gt_5),345).
orientation(wk1gt_1,149,346). coords(wk1gt_1,143,134,346).  visible(wk1gt_1,346). happensAt(walking(wk1gt_1),346). orientation(wk1gt_5,162,346). coords(wk1gt_5,50,163,346).  visible(wk1gt_5,346). happensAt(inactive(wk1gt_5),346).
orientation(wk1gt_1,149,347). coords(wk1gt_1,141,133,347).  visible(wk1gt_1,347). happensAt(walking(wk1gt_1),347). orientation(wk1gt_5,162,347). coords(wk1gt_5,50,163,347).  visible(wk1gt_5,347). happensAt(inactive(wk1gt_5),347).
orientation(wk1gt_1,149,348). coords(wk1gt_1,140,133,348).  visible(wk1gt_1,348). happensAt(walking(wk1gt_1),348). orientation(wk1gt_5,162,348). coords(wk1gt_5,50,163,348).  visible(wk1gt_5,348). happensAt(inactive(wk1gt_5),348).
orientation(wk1gt_1,149,349). coords(wk1gt_1,139,133,349).  visible(wk1gt_1,349). happensAt(walking(wk1gt_1),349). orientation(wk1gt_5,162,349). coords(wk1gt_5,50,163,349).  visible(wk1gt_5,349). happensAt(inactive(wk1gt_5),349).
orientation(wk1gt_1,149,350). coords(wk1gt_1,139,132,350).  visible(wk1gt_1,350). happensAt(walking(wk1gt_1),350). orientation(wk1gt_5,162,350). coords(wk1gt_5,50,163,350).  visible(wk1gt_5,350). happensAt(inactive(wk1gt_5),350).
orientation(wk1gt_1,149,351). coords(wk1gt_1,139,132,351).  visible(wk1gt_1,351). happensAt(walking(wk1gt_1),351). orientation(wk1gt_5,162,351). coords(wk1gt_5,50,163,351).  visible(wk1gt_5,351). happensAt(inactive(wk1gt_5),351).
orientation(wk1gt_1,149,352). coords(wk1gt_1,138,132,352).  visible(wk1gt_1,352). happensAt(walking(wk1gt_1),352). orientation(wk1gt_5,162,352). coords(wk1gt_5,50,163,352).  visible(wk1gt_5,352). happensAt(inactive(wk1gt_5),352).
orientation(wk1gt_1,149,353). coords(wk1gt_1,138,131,353).  visible(wk1gt_1,353). happensAt(walking(wk1gt_1),353). orientation(wk1gt_5,162,353). coords(wk1gt_5,50,163,353).  visible(wk1gt_5,353). happensAt(inactive(wk1gt_5),353).
orientation(wk1gt_1,149,354). coords(wk1gt_1,138,131,354).  visible(wk1gt_1,354). happensAt(walking(wk1gt_1),354). orientation(wk1gt_5,162,354). coords(wk1gt_5,50,163,354).  visible(wk1gt_5,354). happensAt(inactive(wk1gt_5),354).
orientation(wk1gt_1,149,355). coords(wk1gt_1,136,130,355).  visible(wk1gt_1,355). happensAt(walking(wk1gt_1),355). orientation(wk1gt_5,162,355). coords(wk1gt_5,50,163,355).  visible(wk1gt_5,355). happensAt(inactive(wk1gt_5),355).
orientation(wk1gt_1,149,356). coords(wk1gt_1,136,130,356).  visible(wk1gt_1,356). happensAt(walking(wk1gt_1),356). orientation(wk1gt_5,162,356). coords(wk1gt_5,50,163,356).  visible(wk1gt_5,356). happensAt(inactive(wk1gt_5),356).
#show holdsAt/2.