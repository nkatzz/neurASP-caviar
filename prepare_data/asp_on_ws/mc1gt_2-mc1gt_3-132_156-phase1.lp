
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

	orientation(mc1gt_2,144,132). coords(mc1gt_2,76,88,132).  visible(mc1gt_2,132). happensAt(walking(mc1gt_2),132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132). happensAt(walking(mc1gt_3),132).
orientation(mc1gt_2,144,133). coords(mc1gt_2,76,89,133).  visible(mc1gt_2,133). happensAt(walking(mc1gt_2),133). orientation(mc1gt_3,146,133). coords(mc1gt_3,87,79,133).  visible(mc1gt_3,133). happensAt(walking(mc1gt_3),133).
orientation(mc1gt_2,144,134). coords(mc1gt_2,76,89,134).  visible(mc1gt_2,134). happensAt(walking(mc1gt_2),134). orientation(mc1gt_3,146,134). coords(mc1gt_3,89,79,134).  visible(mc1gt_3,134). happensAt(walking(mc1gt_3),134).
orientation(mc1gt_2,144,135). coords(mc1gt_2,76,90,135).  visible(mc1gt_2,135). happensAt(walking(mc1gt_2),135). orientation(mc1gt_3,146,135). coords(mc1gt_3,89,78,135).  visible(mc1gt_3,135). happensAt(walking(mc1gt_3),135).
orientation(mc1gt_2,144,136). coords(mc1gt_2,76,90,136).  visible(mc1gt_2,136). happensAt(walking(mc1gt_2),136). orientation(mc1gt_3,146,136). coords(mc1gt_3,89,78,136).  visible(mc1gt_3,136). happensAt(walking(mc1gt_3),136).
orientation(mc1gt_2,144,137). coords(mc1gt_2,77,90,137).  visible(mc1gt_2,137). happensAt(walking(mc1gt_2),137). orientation(mc1gt_3,146,137). coords(mc1gt_3,90,78,137).  visible(mc1gt_3,137). happensAt(walking(mc1gt_3),137).
orientation(mc1gt_2,144,138). coords(mc1gt_2,77,91,138).  visible(mc1gt_2,138). happensAt(walking(mc1gt_2),138). orientation(mc1gt_3,146,138). coords(mc1gt_3,91,79,138).  visible(mc1gt_3,138). happensAt(walking(mc1gt_3),138).
orientation(mc1gt_2,144,139). coords(mc1gt_2,77,92,139).  visible(mc1gt_2,139). happensAt(walking(mc1gt_2),139). orientation(mc1gt_3,146,139). coords(mc1gt_3,91,79,139).  visible(mc1gt_3,139). happensAt(walking(mc1gt_3),139).
orientation(mc1gt_2,144,140). coords(mc1gt_2,76,92,140).  visible(mc1gt_2,140). happensAt(walking(mc1gt_2),140). orientation(mc1gt_3,146,140). coords(mc1gt_3,91,79,140).  visible(mc1gt_3,140). happensAt(walking(mc1gt_3),140).
orientation(mc1gt_2,144,141). coords(mc1gt_2,75,93,141).  visible(mc1gt_2,141). happensAt(walking(mc1gt_2),141). orientation(mc1gt_3,146,141). coords(mc1gt_3,91,79,141).  visible(mc1gt_3,141). happensAt(walking(mc1gt_3),141).
orientation(mc1gt_2,144,142). coords(mc1gt_2,75,94,142).  visible(mc1gt_2,142). happensAt(walking(mc1gt_2),142). orientation(mc1gt_3,146,142). coords(mc1gt_3,91,79,142).  visible(mc1gt_3,142). happensAt(walking(mc1gt_3),142).
orientation(mc1gt_2,144,143). coords(mc1gt_2,73,95,143).  visible(mc1gt_2,143). happensAt(walking(mc1gt_2),143). orientation(mc1gt_3,146,143). coords(mc1gt_3,91,79,143).  visible(mc1gt_3,143). happensAt(walking(mc1gt_3),143).
orientation(mc1gt_2,144,144). coords(mc1gt_2,73,95,144).  visible(mc1gt_2,144). happensAt(walking(mc1gt_2),144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144). happensAt(walking(mc1gt_3),144).
orientation(mc1gt_2,144,145). coords(mc1gt_2,74,95,145).  visible(mc1gt_2,145). happensAt(walking(mc1gt_2),145). orientation(mc1gt_3,146,145). coords(mc1gt_3,92,79,145).  visible(mc1gt_3,145). happensAt(walking(mc1gt_3),145).
orientation(mc1gt_2,144,146). coords(mc1gt_2,74,95,146).  visible(mc1gt_2,146). happensAt(walking(mc1gt_2),146). orientation(mc1gt_3,146,146). coords(mc1gt_3,94,79,146).  visible(mc1gt_3,146). happensAt(walking(mc1gt_3),146).
orientation(mc1gt_2,144,147). coords(mc1gt_2,74,96,147).  visible(mc1gt_2,147). happensAt(walking(mc1gt_2),147). orientation(mc1gt_3,146,147). coords(mc1gt_3,95,79,147).  visible(mc1gt_3,147). happensAt(walking(mc1gt_3),147).
orientation(mc1gt_2,144,148). coords(mc1gt_2,75,96,148).  visible(mc1gt_2,148). happensAt(walking(mc1gt_2),148). orientation(mc1gt_3,146,148). coords(mc1gt_3,95,79,148).  visible(mc1gt_3,148). happensAt(walking(mc1gt_3),148).
orientation(mc1gt_2,144,149). coords(mc1gt_2,75,98,149).  visible(mc1gt_2,149). happensAt(walking(mc1gt_2),149). orientation(mc1gt_3,146,149). coords(mc1gt_3,96,79,149).  visible(mc1gt_3,149). happensAt(walking(mc1gt_3),149).
orientation(mc1gt_2,144,150). coords(mc1gt_2,78,101,150).  visible(mc1gt_2,150). happensAt(walking(mc1gt_2),150). orientation(mc1gt_3,146,150). coords(mc1gt_3,96,79,150).  visible(mc1gt_3,150). happensAt(walking(mc1gt_3),150).
orientation(mc1gt_2,144,151). coords(mc1gt_2,78,102,151).  visible(mc1gt_2,151). happensAt(walking(mc1gt_2),151). orientation(mc1gt_3,146,151). coords(mc1gt_3,97,79,151).  visible(mc1gt_3,151). happensAt(walking(mc1gt_3),151).
orientation(mc1gt_2,144,152). coords(mc1gt_2,79,103,152).  visible(mc1gt_2,152). happensAt(walking(mc1gt_2),152). orientation(mc1gt_3,146,152). coords(mc1gt_3,98,79,152).  visible(mc1gt_3,152). happensAt(walking(mc1gt_3),152).
orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153). happensAt(walking(mc1gt_2),153). orientation(mc1gt_3,146,153). coords(mc1gt_3,98,79,153).  visible(mc1gt_3,153). happensAt(walking(mc1gt_3),153).
orientation(mc1gt_2,144,154). coords(mc1gt_2,80,105,154).  visible(mc1gt_2,154). happensAt(walking(mc1gt_2),154). orientation(mc1gt_3,146,154). coords(mc1gt_3,99,79,154).  visible(mc1gt_3,154). happensAt(walking(mc1gt_3),154).
orientation(mc1gt_2,144,155). coords(mc1gt_2,80,105,155).  visible(mc1gt_2,155). happensAt(walking(mc1gt_2),155). orientation(mc1gt_3,146,155). coords(mc1gt_3,99,79,155).  visible(mc1gt_3,155). happensAt(walking(mc1gt_3),155).
orientation(mc1gt_2,144,156). coords(mc1gt_2,80,105,156).  visible(mc1gt_2,156). happensAt(walking(mc1gt_2),156). orientation(mc1gt_3,146,156). coords(mc1gt_3,99,80,156).  visible(mc1gt_3,156). happensAt(walking(mc1gt_3),156).
#show holdsAt/2.