
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

	orientation(lb1gt_0,142,132). coords(lb1gt_0,177,111,132).  visible(lb1gt_0,132). happensAt(walking(lb1gt_0),132). orientation(lb1gt_2,144,132). coords(lb1gt_2,96,99,132).  visible(lb1gt_2,132). happensAt(walking(lb1gt_2),132).
orientation(lb1gt_0,142,133). coords(lb1gt_0,176,110,133).  visible(lb1gt_0,133). happensAt(walking(lb1gt_0),133). orientation(lb1gt_2,144,133). coords(lb1gt_2,97,99,133).  visible(lb1gt_2,133). happensAt(walking(lb1gt_2),133).
orientation(lb1gt_0,142,134). coords(lb1gt_0,175,109,134).  visible(lb1gt_0,134). happensAt(walking(lb1gt_0),134). orientation(lb1gt_2,144,134). coords(lb1gt_2,97,99,134).  visible(lb1gt_2,134). happensAt(walking(lb1gt_2),134).
orientation(lb1gt_0,142,135). coords(lb1gt_0,173,108,135).  visible(lb1gt_0,135). happensAt(walking(lb1gt_0),135). orientation(lb1gt_2,144,135). coords(lb1gt_2,98,100,135).  visible(lb1gt_2,135). happensAt(walking(lb1gt_2),135).
orientation(lb1gt_0,142,136). coords(lb1gt_0,173,107,136).  visible(lb1gt_0,136). happensAt(walking(lb1gt_0),136). orientation(lb1gt_2,144,136). coords(lb1gt_2,98,100,136).  visible(lb1gt_2,136). happensAt(walking(lb1gt_2),136).
orientation(lb1gt_0,142,137). coords(lb1gt_0,173,106,137).  visible(lb1gt_0,137). happensAt(walking(lb1gt_0),137). orientation(lb1gt_2,144,137). coords(lb1gt_2,99,101,137).  visible(lb1gt_2,137). happensAt(walking(lb1gt_2),137).
orientation(lb1gt_0,142,138). coords(lb1gt_0,173,104,138).  visible(lb1gt_0,138). happensAt(walking(lb1gt_0),138). orientation(lb1gt_2,144,138). coords(lb1gt_2,99,102,138).  visible(lb1gt_2,138). happensAt(walking(lb1gt_2),138).
orientation(lb1gt_0,138,139). coords(lb1gt_0,172,101,139).  visible(lb1gt_0,139). happensAt(walking(lb1gt_0),139). orientation(lb1gt_2,144,139). coords(lb1gt_2,100,102,139).  visible(lb1gt_2,139). happensAt(walking(lb1gt_2),139).
orientation(lb1gt_0,139,140). coords(lb1gt_0,172,98,140).  visible(lb1gt_0,140). happensAt(walking(lb1gt_0),140). orientation(lb1gt_2,144,140). coords(lb1gt_2,100,102,140).  visible(lb1gt_2,140). happensAt(walking(lb1gt_2),140).
orientation(lb1gt_0,138,141). coords(lb1gt_0,171,98,141).  visible(lb1gt_0,141). happensAt(walking(lb1gt_0),141). orientation(lb1gt_2,144,141). coords(lb1gt_2,100,102,141).  visible(lb1gt_2,141). happensAt(walking(lb1gt_2),141).
orientation(lb1gt_0,136,142). coords(lb1gt_0,171,96,142).  visible(lb1gt_0,142). happensAt(walking(lb1gt_0),142). orientation(lb1gt_2,144,142). coords(lb1gt_2,100,102,142).  visible(lb1gt_2,142). happensAt(walking(lb1gt_2),142).
orientation(lb1gt_0,136,143). coords(lb1gt_0,170,95,143).  visible(lb1gt_0,143). happensAt(walking(lb1gt_0),143). orientation(lb1gt_2,144,143). coords(lb1gt_2,100,102,143).  visible(lb1gt_2,143). happensAt(walking(lb1gt_2),143).
orientation(lb1gt_0,136,144). coords(lb1gt_0,170,93,144).  visible(lb1gt_0,144). happensAt(walking(lb1gt_0),144). orientation(lb1gt_2,144,144). coords(lb1gt_2,100,102,144).  visible(lb1gt_2,144). happensAt(walking(lb1gt_2),144).
orientation(lb1gt_0,136,145). coords(lb1gt_0,169,92,145).  visible(lb1gt_0,145). happensAt(walking(lb1gt_0),145). orientation(lb1gt_2,144,145). coords(lb1gt_2,100,103,145).  visible(lb1gt_2,145). happensAt(walking(lb1gt_2),145).
orientation(lb1gt_0,136,146). coords(lb1gt_0,169,90,146).  visible(lb1gt_0,146). happensAt(walking(lb1gt_0),146). orientation(lb1gt_2,144,146). coords(lb1gt_2,100,104,146).  visible(lb1gt_2,146). happensAt(walking(lb1gt_2),146).
orientation(lb1gt_0,136,147). coords(lb1gt_0,169,88,147).  visible(lb1gt_0,147). happensAt(walking(lb1gt_0),147). orientation(lb1gt_2,144,147). coords(lb1gt_2,100,104,147).  visible(lb1gt_2,147). happensAt(walking(lb1gt_2),147).
orientation(lb1gt_0,136,148). coords(lb1gt_0,168,85,148).  visible(lb1gt_0,148). happensAt(walking(lb1gt_0),148). orientation(lb1gt_2,144,148). coords(lb1gt_2,101,106,148).  visible(lb1gt_2,148). happensAt(walking(lb1gt_2),148).
orientation(lb1gt_0,136,149). coords(lb1gt_0,168,82,149).  visible(lb1gt_0,149). happensAt(walking(lb1gt_0),149). orientation(lb1gt_2,144,149). coords(lb1gt_2,101,106,149).  visible(lb1gt_2,149). happensAt(walking(lb1gt_2),149).
orientation(lb1gt_0,136,150). coords(lb1gt_0,166,80,150).  visible(lb1gt_0,150). happensAt(walking(lb1gt_0),150). orientation(lb1gt_2,144,150). coords(lb1gt_2,101,107,150).  visible(lb1gt_2,150). happensAt(walking(lb1gt_2),150).
orientation(lb1gt_0,136,151). coords(lb1gt_0,163,79,151).  visible(lb1gt_0,151). happensAt(walking(lb1gt_0),151). orientation(lb1gt_2,144,151). coords(lb1gt_2,102,108,151).  visible(lb1gt_2,151). happensAt(walking(lb1gt_2),151).
orientation(lb1gt_0,136,152). coords(lb1gt_0,162,77,152).  visible(lb1gt_0,152). happensAt(walking(lb1gt_0),152). orientation(lb1gt_2,144,152). coords(lb1gt_2,102,108,152).  visible(lb1gt_2,152). happensAt(walking(lb1gt_2),152).
orientation(lb1gt_0,136,153). coords(lb1gt_0,162,76,153).  visible(lb1gt_0,153). happensAt(walking(lb1gt_0),153). orientation(lb1gt_2,144,153). coords(lb1gt_2,103,109,153).  visible(lb1gt_2,153). happensAt(walking(lb1gt_2),153).
orientation(lb1gt_0,136,154). coords(lb1gt_0,162,73,154).  visible(lb1gt_0,154). happensAt(walking(lb1gt_0),154). orientation(lb1gt_2,144,154). coords(lb1gt_2,103,109,154).  visible(lb1gt_2,154). happensAt(walking(lb1gt_2),154).
orientation(lb1gt_0,136,155). coords(lb1gt_0,162,72,155).  visible(lb1gt_0,155). happensAt(walking(lb1gt_0),155). orientation(lb1gt_2,144,155). coords(lb1gt_2,103,109,155).  visible(lb1gt_2,155). happensAt(walking(lb1gt_2),155).
orientation(lb1gt_0,136,156). coords(lb1gt_0,161,71,156).  visible(lb1gt_0,156). happensAt(walking(lb1gt_0),156). orientation(lb1gt_2,144,156). coords(lb1gt_2,103,109,156).  visible(lb1gt_2,156). happensAt(walking(lb1gt_2),156).
#show holdsAt/2.