
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

	orientation(br1gt_2,169,334). coords(br1gt_2,86,207,334).  visible(br1gt_2,334). happensAt(inactive(br1gt_2),334). orientation(br1gt_3,175,334). coords(br1gt_3,26,164,334).  happensAt(appear(br1gt_3),334). happensAt(walking(br1gt_3),334).
orientation(br1gt_2,169,335). coords(br1gt_2,86,207,335).  visible(br1gt_2,335). happensAt(inactive(br1gt_2),335). orientation(br1gt_3,175,335). coords(br1gt_3,26,164,335).  visible(br1gt_3,335). happensAt(walking(br1gt_3),335).
orientation(br1gt_2,169,336). coords(br1gt_2,86,207,336).  visible(br1gt_2,336). happensAt(inactive(br1gt_2),336). orientation(br1gt_3,175,336). coords(br1gt_3,26,164,336).  visible(br1gt_3,336). happensAt(walking(br1gt_3),336).
orientation(br1gt_2,169,337). coords(br1gt_2,86,207,337).  visible(br1gt_2,337). happensAt(inactive(br1gt_2),337). orientation(br1gt_3,175,337). coords(br1gt_3,26,164,337).  visible(br1gt_3,337). happensAt(walking(br1gt_3),337).
orientation(br1gt_2,169,338). coords(br1gt_2,86,207,338).  visible(br1gt_2,338). happensAt(inactive(br1gt_2),338). orientation(br1gt_3,175,338). coords(br1gt_3,26,164,338).  visible(br1gt_3,338). happensAt(walking(br1gt_3),338).
orientation(br1gt_2,169,339). coords(br1gt_2,86,207,339).  visible(br1gt_2,339). happensAt(inactive(br1gt_2),339). orientation(br1gt_3,175,339). coords(br1gt_3,26,164,339).  visible(br1gt_3,339). happensAt(walking(br1gt_3),339).
orientation(br1gt_2,169,340). coords(br1gt_2,86,207,340).  visible(br1gt_2,340). happensAt(inactive(br1gt_2),340). orientation(br1gt_3,175,340). coords(br1gt_3,26,164,340).  visible(br1gt_3,340). happensAt(walking(br1gt_3),340).
orientation(br1gt_2,169,341). coords(br1gt_2,86,207,341).  visible(br1gt_2,341). happensAt(inactive(br1gt_2),341). orientation(br1gt_3,175,341). coords(br1gt_3,26,164,341).  visible(br1gt_3,341). happensAt(walking(br1gt_3),341).
orientation(br1gt_2,169,342). coords(br1gt_2,86,207,342).  visible(br1gt_2,342). happensAt(inactive(br1gt_2),342). orientation(br1gt_3,175,342). coords(br1gt_3,26,164,342).  visible(br1gt_3,342). happensAt(walking(br1gt_3),342).
orientation(br1gt_2,169,343). coords(br1gt_2,86,207,343).  visible(br1gt_2,343). happensAt(inactive(br1gt_2),343). orientation(br1gt_3,175,343). coords(br1gt_3,27,163,343).  visible(br1gt_3,343). happensAt(walking(br1gt_3),343).
orientation(br1gt_2,169,344). coords(br1gt_2,86,207,344).  visible(br1gt_2,344). happensAt(inactive(br1gt_2),344). orientation(br1gt_3,175,344). coords(br1gt_3,27,163,344).  visible(br1gt_3,344). happensAt(walking(br1gt_3),344).
orientation(br1gt_2,169,345). coords(br1gt_2,86,207,345).  visible(br1gt_2,345). happensAt(inactive(br1gt_2),345). orientation(br1gt_3,175,345). coords(br1gt_3,27,163,345).  visible(br1gt_3,345). happensAt(walking(br1gt_3),345).
orientation(br1gt_2,169,346). coords(br1gt_2,86,207,346).  visible(br1gt_2,346). happensAt(inactive(br1gt_2),346). orientation(br1gt_3,175,346). coords(br1gt_3,28,163,346).  visible(br1gt_3,346). happensAt(walking(br1gt_3),346).
orientation(br1gt_2,169,347). coords(br1gt_2,86,207,347).  visible(br1gt_2,347). happensAt(inactive(br1gt_2),347). orientation(br1gt_3,175,347). coords(br1gt_3,28,163,347).  visible(br1gt_3,347). happensAt(walking(br1gt_3),347).
orientation(br1gt_2,169,348). coords(br1gt_2,86,207,348).  visible(br1gt_2,348). happensAt(inactive(br1gt_2),348). orientation(br1gt_3,175,348). coords(br1gt_3,29,163,348).  visible(br1gt_3,348). happensAt(walking(br1gt_3),348).
orientation(br1gt_2,169,349). coords(br1gt_2,86,207,349).  visible(br1gt_2,349). happensAt(inactive(br1gt_2),349). orientation(br1gt_3,175,349). coords(br1gt_3,29,163,349).  visible(br1gt_3,349). happensAt(walking(br1gt_3),349).
orientation(br1gt_2,169,350). coords(br1gt_2,86,207,350).  visible(br1gt_2,350). happensAt(inactive(br1gt_2),350). orientation(br1gt_3,162,350). coords(br1gt_3,29,163,350).  visible(br1gt_3,350). happensAt(walking(br1gt_3),350).
orientation(br1gt_2,169,351). coords(br1gt_2,86,207,351).  visible(br1gt_2,351). happensAt(inactive(br1gt_2),351). orientation(br1gt_3,162,351). coords(br1gt_3,29,163,351).  visible(br1gt_3,351). happensAt(walking(br1gt_3),351).
orientation(br1gt_2,169,352). coords(br1gt_2,86,207,352).  visible(br1gt_2,352). happensAt(inactive(br1gt_2),352). orientation(br1gt_3,162,352). coords(br1gt_3,29,163,352).  visible(br1gt_3,352). happensAt(walking(br1gt_3),352).
orientation(br1gt_2,169,353). coords(br1gt_2,86,207,353).  visible(br1gt_2,353). happensAt(inactive(br1gt_2),353). orientation(br1gt_3,162,353). coords(br1gt_3,29,163,353).  visible(br1gt_3,353). happensAt(walking(br1gt_3),353).
orientation(br1gt_2,169,354). coords(br1gt_2,86,207,354).  visible(br1gt_2,354). happensAt(inactive(br1gt_2),354). orientation(br1gt_3,162,354). coords(br1gt_3,29,163,354).  visible(br1gt_3,354). happensAt(walking(br1gt_3),354).
orientation(br1gt_2,169,355). coords(br1gt_2,86,207,355).  visible(br1gt_2,355). happensAt(inactive(br1gt_2),355). orientation(br1gt_3,162,355). coords(br1gt_3,29,163,355).  visible(br1gt_3,355). happensAt(walking(br1gt_3),355).
orientation(br1gt_2,169,356). coords(br1gt_2,86,207,356).  visible(br1gt_2,356). happensAt(inactive(br1gt_2),356). orientation(br1gt_3,162,356). coords(br1gt_3,30,163,356).  visible(br1gt_3,356). happensAt(walking(br1gt_3),356).
orientation(br1gt_2,169,357). coords(br1gt_2,86,207,357).  visible(br1gt_2,357). happensAt(inactive(br1gt_2),357). orientation(br1gt_3,162,357). coords(br1gt_3,30,163,357).  visible(br1gt_3,357). happensAt(walking(br1gt_3),357).
orientation(br1gt_2,169,358). coords(br1gt_2,86,207,358).  visible(br1gt_2,358). happensAt(inactive(br1gt_2),358). orientation(br1gt_3,162,358). coords(br1gt_3,30,163,358).  visible(br1gt_3,358). happensAt(walking(br1gt_3),358).
#show holdsAt/2.