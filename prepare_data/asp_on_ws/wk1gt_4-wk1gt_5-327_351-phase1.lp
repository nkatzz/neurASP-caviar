
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

	orientation(wk1gt_4,162,327). coords(wk1gt_4,52,176,327).  visible(wk1gt_4,327). happensAt(inactive(wk1gt_4),327). orientation(wk1gt_5,162,327). coords(wk1gt_5,50,163,327).  visible(wk1gt_5,327). happensAt(inactive(wk1gt_5),327).
orientation(wk1gt_4,162,328). coords(wk1gt_4,52,176,328).  visible(wk1gt_4,328). happensAt(inactive(wk1gt_4),328). orientation(wk1gt_5,162,328). coords(wk1gt_5,51,163,328).  visible(wk1gt_5,328). happensAt(inactive(wk1gt_5),328).
orientation(wk1gt_4,158,329). coords(wk1gt_4,52,176,329).  visible(wk1gt_4,329). happensAt(inactive(wk1gt_4),329). orientation(wk1gt_5,162,329). coords(wk1gt_5,51,163,329).  visible(wk1gt_5,329). happensAt(inactive(wk1gt_5),329).
orientation(wk1gt_4,160,330). coords(wk1gt_4,52,176,330).  visible(wk1gt_4,330). happensAt(inactive(wk1gt_4),330). orientation(wk1gt_5,162,330). coords(wk1gt_5,51,163,330).  visible(wk1gt_5,330). happensAt(inactive(wk1gt_5),330).
orientation(wk1gt_4,157,331). coords(wk1gt_4,52,176,331).  visible(wk1gt_4,331). happensAt(inactive(wk1gt_4),331). orientation(wk1gt_5,162,331). coords(wk1gt_5,51,163,331).  visible(wk1gt_5,331). happensAt(inactive(wk1gt_5),331).
orientation(wk1gt_4,160,332). coords(wk1gt_4,52,176,332).  visible(wk1gt_4,332). happensAt(inactive(wk1gt_4),332). orientation(wk1gt_5,162,332). coords(wk1gt_5,50,163,332).  visible(wk1gt_5,332). happensAt(inactive(wk1gt_5),332).
orientation(wk1gt_4,157,333). coords(wk1gt_4,52,176,333).  visible(wk1gt_4,333). happensAt(inactive(wk1gt_4),333). orientation(wk1gt_5,162,333). coords(wk1gt_5,50,163,333).  visible(wk1gt_5,333). happensAt(inactive(wk1gt_5),333).
orientation(wk1gt_4,161,334). coords(wk1gt_4,52,176,334).  visible(wk1gt_4,334). happensAt(inactive(wk1gt_4),334). orientation(wk1gt_5,162,334). coords(wk1gt_5,50,163,334).  visible(wk1gt_5,334). happensAt(inactive(wk1gt_5),334).
orientation(wk1gt_4,165,335). coords(wk1gt_4,52,176,335).  visible(wk1gt_4,335). happensAt(inactive(wk1gt_4),335). orientation(wk1gt_5,162,335). coords(wk1gt_5,50,163,335).  visible(wk1gt_5,335). happensAt(inactive(wk1gt_5),335).
orientation(wk1gt_4,164,336). coords(wk1gt_4,52,176,336).  visible(wk1gt_4,336). happensAt(inactive(wk1gt_4),336). orientation(wk1gt_5,162,336). coords(wk1gt_5,50,163,336).  visible(wk1gt_5,336). happensAt(inactive(wk1gt_5),336).
orientation(wk1gt_4,156,337). coords(wk1gt_4,52,176,337).  visible(wk1gt_4,337). happensAt(inactive(wk1gt_4),337). orientation(wk1gt_5,162,337). coords(wk1gt_5,50,163,337).  visible(wk1gt_5,337). happensAt(inactive(wk1gt_5),337).
orientation(wk1gt_4,163,338). coords(wk1gt_4,52,176,338).  visible(wk1gt_4,338). happensAt(inactive(wk1gt_4),338). orientation(wk1gt_5,162,338). coords(wk1gt_5,50,163,338).  visible(wk1gt_5,338). happensAt(inactive(wk1gt_5),338).
orientation(wk1gt_4,160,339). coords(wk1gt_4,52,176,339).  visible(wk1gt_4,339). happensAt(inactive(wk1gt_4),339). orientation(wk1gt_5,162,339). coords(wk1gt_5,50,163,339).  visible(wk1gt_5,339). happensAt(inactive(wk1gt_5),339).
orientation(wk1gt_4,158,340). coords(wk1gt_4,52,176,340).  visible(wk1gt_4,340). happensAt(inactive(wk1gt_4),340). orientation(wk1gt_5,162,340). coords(wk1gt_5,50,163,340).  visible(wk1gt_5,340). happensAt(inactive(wk1gt_5),340).
orientation(wk1gt_4,162,341). coords(wk1gt_4,52,176,341).  visible(wk1gt_4,341). happensAt(inactive(wk1gt_4),341). orientation(wk1gt_5,162,341). coords(wk1gt_5,50,163,341).  visible(wk1gt_5,341). happensAt(inactive(wk1gt_5),341).
orientation(wk1gt_4,165,342). coords(wk1gt_4,52,176,342).  visible(wk1gt_4,342). happensAt(inactive(wk1gt_4),342). orientation(wk1gt_5,162,342). coords(wk1gt_5,50,163,342).  visible(wk1gt_5,342). happensAt(inactive(wk1gt_5),342).
orientation(wk1gt_4,162,343). coords(wk1gt_4,52,176,343).  visible(wk1gt_4,343). happensAt(inactive(wk1gt_4),343). orientation(wk1gt_5,162,343). coords(wk1gt_5,50,163,343).  visible(wk1gt_5,343). happensAt(inactive(wk1gt_5),343).
orientation(wk1gt_4,167,344). coords(wk1gt_4,52,176,344).  visible(wk1gt_4,344). happensAt(inactive(wk1gt_4),344). orientation(wk1gt_5,162,344). coords(wk1gt_5,50,163,344).  visible(wk1gt_5,344). happensAt(inactive(wk1gt_5),344).
orientation(wk1gt_4,164,345). coords(wk1gt_4,52,176,345).  visible(wk1gt_4,345). happensAt(inactive(wk1gt_4),345). orientation(wk1gt_5,162,345). coords(wk1gt_5,50,163,345).  visible(wk1gt_5,345). happensAt(inactive(wk1gt_5),345).
orientation(wk1gt_4,168,346). coords(wk1gt_4,52,176,346).  visible(wk1gt_4,346). happensAt(inactive(wk1gt_4),346). orientation(wk1gt_5,162,346). coords(wk1gt_5,50,163,346).  visible(wk1gt_5,346). happensAt(inactive(wk1gt_5),346).
orientation(wk1gt_4,168,347). coords(wk1gt_4,52,176,347).  visible(wk1gt_4,347). happensAt(inactive(wk1gt_4),347). orientation(wk1gt_5,162,347). coords(wk1gt_5,50,163,347).  visible(wk1gt_5,347). happensAt(inactive(wk1gt_5),347).
orientation(wk1gt_4,166,348). coords(wk1gt_4,52,176,348).  visible(wk1gt_4,348). happensAt(inactive(wk1gt_4),348). orientation(wk1gt_5,162,348). coords(wk1gt_5,50,163,348).  visible(wk1gt_5,348). happensAt(inactive(wk1gt_5),348).
orientation(wk1gt_4,165,349). coords(wk1gt_4,52,176,349).  visible(wk1gt_4,349). happensAt(inactive(wk1gt_4),349). orientation(wk1gt_5,162,349). coords(wk1gt_5,50,163,349).  visible(wk1gt_5,349). happensAt(inactive(wk1gt_5),349).
orientation(wk1gt_4,165,350). coords(wk1gt_4,52,176,350).  visible(wk1gt_4,350). happensAt(inactive(wk1gt_4),350). orientation(wk1gt_5,162,350). coords(wk1gt_5,50,163,350).  visible(wk1gt_5,350). happensAt(inactive(wk1gt_5),350).
orientation(wk1gt_4,167,351). coords(wk1gt_4,52,176,351).  visible(wk1gt_4,351). happensAt(inactive(wk1gt_4),351). orientation(wk1gt_5,162,351). coords(wk1gt_5,50,163,351).  visible(wk1gt_5,351). happensAt(inactive(wk1gt_5),351).
#show holdsAt/2.