
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

	orientation(lb1gt_1,151,336). coords(lb1gt_1,165,169,336).  visible(lb1gt_1,336). happensAt(active(lb1gt_1),336). orientation(lb1gt_2,146,336). coords(lb1gt_2,190,150,336).  visible(lb1gt_2,336). happensAt(active(lb1gt_2),336).
orientation(lb1gt_1,151,337). coords(lb1gt_1,165,169,337).  visible(lb1gt_1,337). happensAt(active(lb1gt_1),337). orientation(lb1gt_2,146,337). coords(lb1gt_2,190,150,337).  visible(lb1gt_2,337). happensAt(active(lb1gt_2),337).
orientation(lb1gt_1,151,338). coords(lb1gt_1,165,169,338).  visible(lb1gt_1,338). happensAt(active(lb1gt_1),338). orientation(lb1gt_2,146,338). coords(lb1gt_2,190,150,338).  visible(lb1gt_2,338). happensAt(active(lb1gt_2),338).
orientation(lb1gt_1,151,339). coords(lb1gt_1,165,169,339).  visible(lb1gt_1,339). happensAt(active(lb1gt_1),339). orientation(lb1gt_2,146,339). coords(lb1gt_2,190,149,339).  visible(lb1gt_2,339). happensAt(active(lb1gt_2),339).
orientation(lb1gt_1,151,340). coords(lb1gt_1,165,169,340).  visible(lb1gt_1,340). happensAt(active(lb1gt_1),340). orientation(lb1gt_2,146,340). coords(lb1gt_2,190,149,340).  visible(lb1gt_2,340). happensAt(active(lb1gt_2),340).
orientation(lb1gt_1,151,341). coords(lb1gt_1,165,169,341).  visible(lb1gt_1,341). happensAt(active(lb1gt_1),341). orientation(lb1gt_2,146,341). coords(lb1gt_2,189,149,341).  visible(lb1gt_2,341). happensAt(active(lb1gt_2),341).
orientation(lb1gt_1,151,342). coords(lb1gt_1,165,169,342).  visible(lb1gt_1,342). happensAt(active(lb1gt_1),342). orientation(lb1gt_2,146,342). coords(lb1gt_2,189,149,342).  visible(lb1gt_2,342). happensAt(active(lb1gt_2),342).
orientation(lb1gt_1,151,343). coords(lb1gt_1,165,169,343).  visible(lb1gt_1,343). happensAt(active(lb1gt_1),343). orientation(lb1gt_2,146,343). coords(lb1gt_2,189,149,343).  visible(lb1gt_2,343). happensAt(active(lb1gt_2),343).
orientation(lb1gt_1,151,344). coords(lb1gt_1,165,169,344).  visible(lb1gt_1,344). happensAt(active(lb1gt_1),344). orientation(lb1gt_2,146,344). coords(lb1gt_2,189,148,344).  visible(lb1gt_2,344). happensAt(active(lb1gt_2),344).
orientation(lb1gt_1,151,345). coords(lb1gt_1,165,169,345).  visible(lb1gt_1,345). happensAt(active(lb1gt_1),345). orientation(lb1gt_2,146,345). coords(lb1gt_2,189,148,345).  visible(lb1gt_2,345). happensAt(active(lb1gt_2),345).
orientation(lb1gt_1,151,346). coords(lb1gt_1,165,169,346).  visible(lb1gt_1,346). happensAt(active(lb1gt_1),346). orientation(lb1gt_2,146,346). coords(lb1gt_2,189,148,346).  visible(lb1gt_2,346). happensAt(active(lb1gt_2),346).
orientation(lb1gt_1,151,347). coords(lb1gt_1,165,169,347).  visible(lb1gt_1,347). happensAt(active(lb1gt_1),347). orientation(lb1gt_2,146,347). coords(lb1gt_2,189,148,347).  visible(lb1gt_2,347). happensAt(active(lb1gt_2),347).
orientation(lb1gt_1,151,348). coords(lb1gt_1,165,169,348).  visible(lb1gt_1,348). happensAt(active(lb1gt_1),348). orientation(lb1gt_2,146,348). coords(lb1gt_2,189,148,348).  visible(lb1gt_2,348). happensAt(active(lb1gt_2),348).
orientation(lb1gt_1,151,349). coords(lb1gt_1,165,169,349).  visible(lb1gt_1,349). happensAt(active(lb1gt_1),349). orientation(lb1gt_2,146,349). coords(lb1gt_2,189,148,349).  visible(lb1gt_2,349). happensAt(active(lb1gt_2),349).
orientation(lb1gt_1,151,350). coords(lb1gt_1,165,169,350).  visible(lb1gt_1,350). happensAt(active(lb1gt_1),350). orientation(lb1gt_2,146,350). coords(lb1gt_2,189,148,350).  visible(lb1gt_2,350). happensAt(active(lb1gt_2),350).
orientation(lb1gt_1,151,351). coords(lb1gt_1,165,169,351).  visible(lb1gt_1,351). happensAt(active(lb1gt_1),351). orientation(lb1gt_2,146,351). coords(lb1gt_2,189,148,351).  visible(lb1gt_2,351). happensAt(active(lb1gt_2),351).
orientation(lb1gt_1,151,352). coords(lb1gt_1,165,170,352).  visible(lb1gt_1,352). happensAt(active(lb1gt_1),352). orientation(lb1gt_2,146,352). coords(lb1gt_2,189,148,352).  visible(lb1gt_2,352). happensAt(active(lb1gt_2),352).
orientation(lb1gt_1,151,353). coords(lb1gt_1,165,170,353).  visible(lb1gt_1,353). happensAt(active(lb1gt_1),353). orientation(lb1gt_2,146,353). coords(lb1gt_2,189,148,353).  visible(lb1gt_2,353). happensAt(active(lb1gt_2),353).
orientation(lb1gt_1,151,354). coords(lb1gt_1,165,170,354).  visible(lb1gt_1,354). happensAt(active(lb1gt_1),354). orientation(lb1gt_2,146,354). coords(lb1gt_2,189,148,354).  visible(lb1gt_2,354). happensAt(active(lb1gt_2),354).
orientation(lb1gt_1,151,355). coords(lb1gt_1,165,170,355).  visible(lb1gt_1,355). happensAt(active(lb1gt_1),355). orientation(lb1gt_2,146,355). coords(lb1gt_2,189,148,355).  visible(lb1gt_2,355). happensAt(active(lb1gt_2),355).
orientation(lb1gt_1,151,356). coords(lb1gt_1,165,170,356).  visible(lb1gt_1,356). happensAt(active(lb1gt_1),356). orientation(lb1gt_2,146,356). coords(lb1gt_2,189,147,356).  visible(lb1gt_2,356). happensAt(active(lb1gt_2),356).
orientation(lb1gt_1,151,357). coords(lb1gt_1,165,170,357).  visible(lb1gt_1,357). happensAt(active(lb1gt_1),357). orientation(lb1gt_2,146,357). coords(lb1gt_2,189,147,357).  visible(lb1gt_2,357). happensAt(active(lb1gt_2),357).
orientation(lb1gt_1,151,358). coords(lb1gt_1,165,170,358).  visible(lb1gt_1,358). happensAt(active(lb1gt_1),358). orientation(lb1gt_2,146,358). coords(lb1gt_2,189,145,358).  visible(lb1gt_2,358). happensAt(active(lb1gt_2),358).
orientation(lb1gt_1,151,359). coords(lb1gt_1,165,170,359).  visible(lb1gt_1,359). happensAt(active(lb1gt_1),359). orientation(lb1gt_2,146,359). coords(lb1gt_2,189,145,359).  visible(lb1gt_2,359). happensAt(active(lb1gt_2),359).
orientation(lb1gt_1,151,360). coords(lb1gt_1,165,170,360).  visible(lb1gt_1,360). happensAt(active(lb1gt_1),360). orientation(lb1gt_2,146,360). coords(lb1gt_2,189,145,360).  visible(lb1gt_2,360). happensAt(active(lb1gt_2),360).
#show holdsAt/2.