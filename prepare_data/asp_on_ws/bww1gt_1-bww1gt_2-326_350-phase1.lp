
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

	orientation(bww1gt_1,179,326). coords(bww1gt_1,105,264,326).  visible(bww1gt_1,326). happensAt(inactive(bww1gt_1),326). orientation(bww1gt_2,167,326). coords(bww1gt_2,69,160,326).  visible(bww1gt_2,326). happensAt(walking(bww1gt_2),326).
orientation(bww1gt_1,9,327). coords(bww1gt_1,105,264,327).  visible(bww1gt_1,327). happensAt(inactive(bww1gt_1),327). orientation(bww1gt_2,167,327). coords(bww1gt_2,69,160,327).  visible(bww1gt_2,327). happensAt(walking(bww1gt_2),327).
orientation(bww1gt_1,8,328). coords(bww1gt_1,105,264,328).  visible(bww1gt_1,328). happensAt(inactive(bww1gt_1),328). orientation(bww1gt_2,167,328). coords(bww1gt_2,70,160,328).  visible(bww1gt_2,328). happensAt(walking(bww1gt_2),328).
orientation(bww1gt_1,7,329). coords(bww1gt_1,105,264,329).  visible(bww1gt_1,329). happensAt(inactive(bww1gt_1),329). orientation(bww1gt_2,167,329). coords(bww1gt_2,70,160,329).  visible(bww1gt_2,329). happensAt(walking(bww1gt_2),329).
orientation(bww1gt_1,8,330). coords(bww1gt_1,105,264,330).  visible(bww1gt_1,330). happensAt(inactive(bww1gt_1),330). orientation(bww1gt_2,167,330). coords(bww1gt_2,70,160,330).  visible(bww1gt_2,330). happensAt(walking(bww1gt_2),330).
orientation(bww1gt_1,5,331). coords(bww1gt_1,105,264,331).  visible(bww1gt_1,331). happensAt(inactive(bww1gt_1),331). orientation(bww1gt_2,167,331). coords(bww1gt_2,70,160,331).  visible(bww1gt_2,331). happensAt(walking(bww1gt_2),331).
orientation(bww1gt_1,4,332). coords(bww1gt_1,105,264,332).  visible(bww1gt_1,332). happensAt(inactive(bww1gt_1),332). orientation(bww1gt_2,167,332). coords(bww1gt_2,70,160,332).  visible(bww1gt_2,332). happensAt(walking(bww1gt_2),332).
orientation(bww1gt_1,4,333). coords(bww1gt_1,105,264,333).  visible(bww1gt_1,333). happensAt(inactive(bww1gt_1),333). orientation(bww1gt_2,167,333). coords(bww1gt_2,70,160,333).  visible(bww1gt_2,333). happensAt(walking(bww1gt_2),333).
orientation(bww1gt_1,6,334). coords(bww1gt_1,105,264,334).  visible(bww1gt_1,334). happensAt(inactive(bww1gt_1),334). orientation(bww1gt_2,167,334). coords(bww1gt_2,70,159,334).  visible(bww1gt_2,334). happensAt(walking(bww1gt_2),334).
orientation(bww1gt_1,6,335). coords(bww1gt_1,105,264,335).  visible(bww1gt_1,335). happensAt(inactive(bww1gt_1),335). orientation(bww1gt_2,167,335). coords(bww1gt_2,71,159,335).  visible(bww1gt_2,335). happensAt(walking(bww1gt_2),335).
orientation(bww1gt_1,9,336). coords(bww1gt_1,105,264,336).  visible(bww1gt_1,336). happensAt(inactive(bww1gt_1),336). orientation(bww1gt_2,167,336). coords(bww1gt_2,72,159,336).  visible(bww1gt_2,336). happensAt(walking(bww1gt_2),336).
orientation(bww1gt_1,5,337). coords(bww1gt_1,105,264,337).  visible(bww1gt_1,337). happensAt(inactive(bww1gt_1),337). orientation(bww1gt_2,167,337). coords(bww1gt_2,73,159,337).  visible(bww1gt_2,337). happensAt(walking(bww1gt_2),337).
orientation(bww1gt_1,7,338). coords(bww1gt_1,105,264,338).  visible(bww1gt_1,338). happensAt(inactive(bww1gt_1),338). orientation(bww1gt_2,167,338). coords(bww1gt_2,75,159,338).  visible(bww1gt_2,338). happensAt(walking(bww1gt_2),338).
orientation(bww1gt_1,9,339). coords(bww1gt_1,105,264,339).  visible(bww1gt_1,339). happensAt(inactive(bww1gt_1),339). orientation(bww1gt_2,167,339). coords(bww1gt_2,75,159,339).  visible(bww1gt_2,339). happensAt(walking(bww1gt_2),339).
orientation(bww1gt_1,6,340). coords(bww1gt_1,105,264,340).  visible(bww1gt_1,340). happensAt(inactive(bww1gt_1),340). orientation(bww1gt_2,167,340). coords(bww1gt_2,76,159,340).  visible(bww1gt_2,340). happensAt(walking(bww1gt_2),340).
orientation(bww1gt_1,5,341). coords(bww1gt_1,105,264,341).  visible(bww1gt_1,341). happensAt(inactive(bww1gt_1),341). orientation(bww1gt_2,167,341). coords(bww1gt_2,76,159,341).  visible(bww1gt_2,341). happensAt(walking(bww1gt_2),341).
orientation(bww1gt_1,8,342). coords(bww1gt_1,105,264,342).  visible(bww1gt_1,342). happensAt(inactive(bww1gt_1),342). orientation(bww1gt_2,167,342). coords(bww1gt_2,76,159,342).  visible(bww1gt_2,342). happensAt(walking(bww1gt_2),342).
orientation(bww1gt_1,8,343). coords(bww1gt_1,105,264,343).  visible(bww1gt_1,343). happensAt(inactive(bww1gt_1),343). orientation(bww1gt_2,167,343). coords(bww1gt_2,76,159,343).  visible(bww1gt_2,343). happensAt(walking(bww1gt_2),343).
orientation(bww1gt_1,10,344). coords(bww1gt_1,105,264,344).  visible(bww1gt_1,344). happensAt(inactive(bww1gt_1),344). orientation(bww1gt_2,167,344). coords(bww1gt_2,76,159,344).  visible(bww1gt_2,344). happensAt(walking(bww1gt_2),344).
orientation(bww1gt_1,7,345). coords(bww1gt_1,105,264,345).  visible(bww1gt_1,345). happensAt(inactive(bww1gt_1),345). orientation(bww1gt_2,167,345). coords(bww1gt_2,76,159,345).  visible(bww1gt_2,345). happensAt(walking(bww1gt_2),345).
orientation(bww1gt_1,8,346). coords(bww1gt_1,105,264,346).  visible(bww1gt_1,346). happensAt(inactive(bww1gt_1),346). orientation(bww1gt_2,167,346). coords(bww1gt_2,76,159,346).  visible(bww1gt_2,346). happensAt(walking(bww1gt_2),346).
orientation(bww1gt_1,4,347). coords(bww1gt_1,105,264,347).  visible(bww1gt_1,347). happensAt(inactive(bww1gt_1),347). orientation(bww1gt_2,167,347). coords(bww1gt_2,77,159,347).  visible(bww1gt_2,347). happensAt(walking(bww1gt_2),347).
orientation(bww1gt_1,0,348). coords(bww1gt_1,105,264,348).  visible(bww1gt_1,348). happensAt(inactive(bww1gt_1),348). orientation(bww1gt_2,167,348). coords(bww1gt_2,77,158,348).  visible(bww1gt_2,348). happensAt(walking(bww1gt_2),348).
orientation(bww1gt_1,5,349). coords(bww1gt_1,105,264,349).  visible(bww1gt_1,349). happensAt(inactive(bww1gt_1),349). orientation(bww1gt_2,167,349). coords(bww1gt_2,77,158,349).  visible(bww1gt_2,349). happensAt(walking(bww1gt_2),349).
orientation(bww1gt_1,3,350). coords(bww1gt_1,105,264,350).  visible(bww1gt_1,350). happensAt(inactive(bww1gt_1),350). orientation(bww1gt_2,167,350). coords(bww1gt_2,78,158,350).  visible(bww1gt_2,350). happensAt(walking(bww1gt_2),350).
#show holdsAt/2.