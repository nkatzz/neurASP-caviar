
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

	orientation(mws1gt_3,165,339). coords(mws1gt_3,59,149,339).  visible(mws1gt_3,339). happensAt(walking(mws1gt_3),339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339). happensAt(walking(mws1gt_7),339).
orientation(mws1gt_3,165,340). coords(mws1gt_3,58,150,340).  visible(mws1gt_3,340). happensAt(walking(mws1gt_3),340). orientation(mws1gt_7,167,340). coords(mws1gt_7,41,193,340).  visible(mws1gt_7,340). happensAt(walking(mws1gt_7),340).
orientation(mws1gt_3,165,341). coords(mws1gt_3,57,150,341).  visible(mws1gt_3,341). happensAt(walking(mws1gt_3),341). orientation(mws1gt_7,167,341). coords(mws1gt_7,41,192,341).  visible(mws1gt_7,341). happensAt(walking(mws1gt_7),341).
orientation(mws1gt_3,165,342). coords(mws1gt_3,56,150,342).  visible(mws1gt_3,342). happensAt(walking(mws1gt_3),342). orientation(mws1gt_7,174,342). coords(mws1gt_7,41,191,342).  visible(mws1gt_7,342). happensAt(walking(mws1gt_7),342).
orientation(mws1gt_3,165,343). coords(mws1gt_3,55,151,343).  visible(mws1gt_3,343). happensAt(walking(mws1gt_3),343). orientation(mws1gt_7,174,343). coords(mws1gt_7,42,191,343).  visible(mws1gt_7,343). happensAt(walking(mws1gt_7),343).
orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344). orientation(mws1gt_7,174,344). coords(mws1gt_7,44,190,344).  visible(mws1gt_7,344). happensAt(walking(mws1gt_7),344).
orientation(mws1gt_3,165,345). coords(mws1gt_3,55,151,345).  visible(mws1gt_3,345). happensAt(walking(mws1gt_3),345). orientation(mws1gt_7,174,345). coords(mws1gt_7,45,190,345).  visible(mws1gt_7,345). happensAt(walking(mws1gt_7),345).
orientation(mws1gt_3,165,346). coords(mws1gt_3,54,152,346).  visible(mws1gt_3,346). happensAt(walking(mws1gt_3),346). orientation(mws1gt_7,174,346). coords(mws1gt_7,45,190,346).  visible(mws1gt_7,346). happensAt(walking(mws1gt_7),346).
orientation(mws1gt_3,165,347). coords(mws1gt_3,54,152,347).  visible(mws1gt_3,347). happensAt(walking(mws1gt_3),347). orientation(mws1gt_7,174,347). coords(mws1gt_7,44,190,347).  visible(mws1gt_7,347). happensAt(walking(mws1gt_7),347).
orientation(mws1gt_3,165,348). coords(mws1gt_3,54,152,348).  visible(mws1gt_3,348). happensAt(walking(mws1gt_3),348). orientation(mws1gt_7,174,348). coords(mws1gt_7,45,190,348).  visible(mws1gt_7,348). happensAt(walking(mws1gt_7),348).
orientation(mws1gt_3,165,349). coords(mws1gt_3,54,153,349).  visible(mws1gt_3,349). happensAt(walking(mws1gt_3),349). orientation(mws1gt_7,174,349). coords(mws1gt_7,46,189,349).  visible(mws1gt_7,349). happensAt(walking(mws1gt_7),349).
orientation(mws1gt_3,165,350). coords(mws1gt_3,54,154,350).  visible(mws1gt_3,350). happensAt(walking(mws1gt_3),350). orientation(mws1gt_7,174,350). coords(mws1gt_7,46,188,350).  visible(mws1gt_7,350). happensAt(walking(mws1gt_7),350).
orientation(mws1gt_3,165,351). coords(mws1gt_3,53,154,351).  visible(mws1gt_3,351). happensAt(walking(mws1gt_3),351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351). happensAt(walking(mws1gt_7),351).
orientation(mws1gt_3,165,352). coords(mws1gt_3,53,154,352).  visible(mws1gt_3,352). happensAt(walking(mws1gt_3),352). orientation(mws1gt_7,174,352). coords(mws1gt_7,46,188,352).  visible(mws1gt_7,352). happensAt(walking(mws1gt_7),352).
orientation(mws1gt_3,165,353). coords(mws1gt_3,53,154,353).  visible(mws1gt_3,353). happensAt(walking(mws1gt_3),353). orientation(mws1gt_7,174,353). coords(mws1gt_7,46,188,353).  visible(mws1gt_7,353). happensAt(walking(mws1gt_7),353).
orientation(mws1gt_3,165,354). coords(mws1gt_3,52,154,354).  visible(mws1gt_3,354). happensAt(walking(mws1gt_3),354). orientation(mws1gt_7,174,354). coords(mws1gt_7,46,188,354).  visible(mws1gt_7,354). happensAt(walking(mws1gt_7),354).
orientation(mws1gt_3,165,355). coords(mws1gt_3,51,154,355).  visible(mws1gt_3,355). happensAt(walking(mws1gt_3),355). orientation(mws1gt_7,174,355). coords(mws1gt_7,46,187,355).  visible(mws1gt_7,355). happensAt(walking(mws1gt_7),355).
orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). happensAt(walking(mws1gt_3),356). orientation(mws1gt_7,174,356). coords(mws1gt_7,46,187,356).  visible(mws1gt_7,356). happensAt(walking(mws1gt_7),356).
orientation(mws1gt_3,165,357). coords(mws1gt_3,50,155,357).  visible(mws1gt_3,357). happensAt(walking(mws1gt_3),357). orientation(mws1gt_7,174,357). coords(mws1gt_7,47,187,357).  visible(mws1gt_7,357). happensAt(walking(mws1gt_7),357).
orientation(mws1gt_3,165,358). coords(mws1gt_3,50,157,358).  visible(mws1gt_3,358). happensAt(walking(mws1gt_3),358). orientation(mws1gt_7,174,358). coords(mws1gt_7,47,187,358).  visible(mws1gt_7,358). happensAt(walking(mws1gt_7),358).
orientation(mws1gt_3,165,359). coords(mws1gt_3,49,157,359).  visible(mws1gt_3,359). happensAt(walking(mws1gt_3),359). orientation(mws1gt_7,174,359). coords(mws1gt_7,49,186,359).  visible(mws1gt_7,359). happensAt(walking(mws1gt_7),359).
orientation(mws1gt_3,165,360). coords(mws1gt_3,49,158,360).  visible(mws1gt_3,360). happensAt(walking(mws1gt_3),360). orientation(mws1gt_7,174,360). coords(mws1gt_7,50,185,360).  visible(mws1gt_7,360). happensAt(walking(mws1gt_7),360).
orientation(mws1gt_3,165,361). coords(mws1gt_3,49,159,361).  visible(mws1gt_3,361). happensAt(walking(mws1gt_3),361). orientation(mws1gt_7,174,361). coords(mws1gt_7,50,185,361).  visible(mws1gt_7,361). happensAt(walking(mws1gt_7),361).
orientation(mws1gt_3,165,362). coords(mws1gt_3,49,159,362).  visible(mws1gt_3,362). happensAt(walking(mws1gt_3),362). orientation(mws1gt_7,174,362). coords(mws1gt_7,50,185,362).  visible(mws1gt_7,362). happensAt(walking(mws1gt_7),362).
orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). happensAt(walking(mws1gt_3),363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363). happensAt(walking(mws1gt_7),363).
#show holdsAt/2.