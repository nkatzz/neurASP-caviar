
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

	orientation(lb1gt_1,151,132). coords(lb1gt_1,87,133,132).  visible(lb1gt_1,132). happensAt(walking(lb1gt_1),132). orientation(lb1gt_2,144,132). coords(lb1gt_2,96,99,132).  visible(lb1gt_2,132). happensAt(walking(lb1gt_2),132).
orientation(lb1gt_1,151,133). coords(lb1gt_1,87,133,133).  visible(lb1gt_1,133). happensAt(walking(lb1gt_1),133). orientation(lb1gt_2,144,133). coords(lb1gt_2,97,99,133).  visible(lb1gt_2,133). happensAt(walking(lb1gt_2),133).
orientation(lb1gt_1,151,134). coords(lb1gt_1,87,133,134).  visible(lb1gt_1,134). happensAt(walking(lb1gt_1),134). orientation(lb1gt_2,144,134). coords(lb1gt_2,97,99,134).  visible(lb1gt_2,134). happensAt(walking(lb1gt_2),134).
orientation(lb1gt_1,151,135). coords(lb1gt_1,87,134,135).  visible(lb1gt_1,135). happensAt(walking(lb1gt_1),135). orientation(lb1gt_2,144,135). coords(lb1gt_2,98,100,135).  visible(lb1gt_2,135). happensAt(walking(lb1gt_2),135).
orientation(lb1gt_1,151,136). coords(lb1gt_1,87,134,136).  visible(lb1gt_1,136). happensAt(walking(lb1gt_1),136). orientation(lb1gt_2,144,136). coords(lb1gt_2,98,100,136).  visible(lb1gt_2,136). happensAt(walking(lb1gt_2),136).
orientation(lb1gt_1,151,137). coords(lb1gt_1,87,134,137).  visible(lb1gt_1,137). happensAt(walking(lb1gt_1),137). orientation(lb1gt_2,144,137). coords(lb1gt_2,99,101,137).  visible(lb1gt_2,137). happensAt(walking(lb1gt_2),137).
orientation(lb1gt_1,151,138). coords(lb1gt_1,87,134,138).  visible(lb1gt_1,138). happensAt(walking(lb1gt_1),138). orientation(lb1gt_2,144,138). coords(lb1gt_2,99,102,138).  visible(lb1gt_2,138). happensAt(walking(lb1gt_2),138).
orientation(lb1gt_1,151,139). coords(lb1gt_1,87,134,139).  visible(lb1gt_1,139). happensAt(walking(lb1gt_1),139). orientation(lb1gt_2,144,139). coords(lb1gt_2,100,102,139).  visible(lb1gt_2,139). happensAt(walking(lb1gt_2),139).
orientation(lb1gt_1,151,140). coords(lb1gt_1,87,135,140).  visible(lb1gt_1,140). happensAt(walking(lb1gt_1),140). orientation(lb1gt_2,144,140). coords(lb1gt_2,100,102,140).  visible(lb1gt_2,140). happensAt(walking(lb1gt_2),140).
orientation(lb1gt_1,151,141). coords(lb1gt_1,88,135,141).  visible(lb1gt_1,141). happensAt(walking(lb1gt_1),141). orientation(lb1gt_2,144,141). coords(lb1gt_2,100,102,141).  visible(lb1gt_2,141). happensAt(walking(lb1gt_2),141).
orientation(lb1gt_1,151,142). coords(lb1gt_1,90,135,142).  visible(lb1gt_1,142). happensAt(walking(lb1gt_1),142). orientation(lb1gt_2,144,142). coords(lb1gt_2,100,102,142).  visible(lb1gt_2,142). happensAt(walking(lb1gt_2),142).
orientation(lb1gt_1,151,143). coords(lb1gt_1,90,135,143).  visible(lb1gt_1,143). happensAt(walking(lb1gt_1),143). orientation(lb1gt_2,144,143). coords(lb1gt_2,100,102,143).  visible(lb1gt_2,143). happensAt(walking(lb1gt_2),143).
orientation(lb1gt_1,151,144). coords(lb1gt_1,92,135,144).  visible(lb1gt_1,144). happensAt(walking(lb1gt_1),144). orientation(lb1gt_2,144,144). coords(lb1gt_2,100,102,144).  visible(lb1gt_2,144). happensAt(walking(lb1gt_2),144).
orientation(lb1gt_1,151,145). coords(lb1gt_1,92,136,145).  visible(lb1gt_1,145). happensAt(walking(lb1gt_1),145). orientation(lb1gt_2,144,145). coords(lb1gt_2,100,103,145).  visible(lb1gt_2,145). happensAt(walking(lb1gt_2),145).
orientation(lb1gt_1,151,146). coords(lb1gt_1,92,136,146).  visible(lb1gt_1,146). happensAt(walking(lb1gt_1),146). orientation(lb1gt_2,144,146). coords(lb1gt_2,100,104,146).  visible(lb1gt_2,146). happensAt(walking(lb1gt_2),146).
orientation(lb1gt_1,151,147). coords(lb1gt_1,92,137,147).  visible(lb1gt_1,147). happensAt(walking(lb1gt_1),147). orientation(lb1gt_2,144,147). coords(lb1gt_2,100,104,147).  visible(lb1gt_2,147). happensAt(walking(lb1gt_2),147).
orientation(lb1gt_1,151,148). coords(lb1gt_1,93,138,148).  visible(lb1gt_1,148). happensAt(walking(lb1gt_1),148). orientation(lb1gt_2,144,148). coords(lb1gt_2,101,106,148).  visible(lb1gt_2,148). happensAt(walking(lb1gt_2),148).
orientation(lb1gt_1,151,149). coords(lb1gt_1,94,138,149).  visible(lb1gt_1,149). happensAt(walking(lb1gt_1),149). orientation(lb1gt_2,144,149). coords(lb1gt_2,101,106,149).  visible(lb1gt_2,149). happensAt(walking(lb1gt_2),149).
orientation(lb1gt_1,151,150). coords(lb1gt_1,94,138,150).  visible(lb1gt_1,150). happensAt(walking(lb1gt_1),150). orientation(lb1gt_2,144,150). coords(lb1gt_2,101,107,150).  visible(lb1gt_2,150). happensAt(walking(lb1gt_2),150).
orientation(lb1gt_1,151,151). coords(lb1gt_1,94,138,151).  visible(lb1gt_1,151). happensAt(walking(lb1gt_1),151). orientation(lb1gt_2,144,151). coords(lb1gt_2,102,108,151).  visible(lb1gt_2,151). happensAt(walking(lb1gt_2),151).
orientation(lb1gt_1,151,152). coords(lb1gt_1,94,138,152).  visible(lb1gt_1,152). happensAt(walking(lb1gt_1),152). orientation(lb1gt_2,144,152). coords(lb1gt_2,102,108,152).  visible(lb1gt_2,152). happensAt(walking(lb1gt_2),152).
orientation(lb1gt_1,151,153). coords(lb1gt_1,94,139,153).  visible(lb1gt_1,153). happensAt(walking(lb1gt_1),153). orientation(lb1gt_2,144,153). coords(lb1gt_2,103,109,153).  visible(lb1gt_2,153). happensAt(walking(lb1gt_2),153).
orientation(lb1gt_1,151,154). coords(lb1gt_1,94,139,154).  visible(lb1gt_1,154). happensAt(walking(lb1gt_1),154). orientation(lb1gt_2,144,154). coords(lb1gt_2,103,109,154).  visible(lb1gt_2,154). happensAt(walking(lb1gt_2),154).
orientation(lb1gt_1,151,155). coords(lb1gt_1,95,139,155).  visible(lb1gt_1,155). happensAt(walking(lb1gt_1),155). orientation(lb1gt_2,144,155). coords(lb1gt_2,103,109,155).  visible(lb1gt_2,155). happensAt(walking(lb1gt_2),155).
orientation(lb1gt_1,151,156). coords(lb1gt_1,95,139,156).  visible(lb1gt_1,156). happensAt(walking(lb1gt_1),156). orientation(lb1gt_2,144,156). coords(lb1gt_2,103,109,156).  visible(lb1gt_2,156). happensAt(walking(lb1gt_2),156).
#show holdsAt/2.