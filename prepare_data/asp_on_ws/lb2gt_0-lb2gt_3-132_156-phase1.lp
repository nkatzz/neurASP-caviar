
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

	orientation(lb2gt_0,141,132). coords(lb2gt_0,193,126,132).  visible(lb2gt_0,132). happensAt(active(lb2gt_0),132). orientation(lb2gt_3,156,132). coords(lb2gt_3,7,59,132).  visible(lb2gt_3,132). happensAt(walking(lb2gt_3),132).
orientation(lb2gt_0,141,133). coords(lb2gt_0,192,125,133).  visible(lb2gt_0,133). happensAt(active(lb2gt_0),133). orientation(lb2gt_3,160,133). coords(lb2gt_3,7,59,133).  visible(lb2gt_3,133). happensAt(walking(lb2gt_3),133).
orientation(lb2gt_0,141,134). coords(lb2gt_0,191,125,134).  visible(lb2gt_0,134). happensAt(active(lb2gt_0),134). orientation(lb2gt_3,163,134). coords(lb2gt_3,7,60,134).  visible(lb2gt_3,134). happensAt(walking(lb2gt_3),134).
orientation(lb2gt_0,141,135). coords(lb2gt_0,190,123,135).  visible(lb2gt_0,135). happensAt(active(lb2gt_0),135). orientation(lb2gt_3,162,135). coords(lb2gt_3,7,60,135).  visible(lb2gt_3,135). happensAt(walking(lb2gt_3),135).
orientation(lb2gt_0,141,136). coords(lb2gt_0,189,121,136).  visible(lb2gt_0,136). happensAt(active(lb2gt_0),136). orientation(lb2gt_3,163,136). coords(lb2gt_3,6,60,136).  visible(lb2gt_3,136). happensAt(walking(lb2gt_3),136).
orientation(lb2gt_0,141,137). coords(lb2gt_0,188,120,137).  visible(lb2gt_0,137). happensAt(active(lb2gt_0),137). orientation(lb2gt_3,167,137). coords(lb2gt_3,6,60,137).  visible(lb2gt_3,137). happensAt(walking(lb2gt_3),137).
orientation(lb2gt_0,141,138). coords(lb2gt_0,187,120,138).  visible(lb2gt_0,138). happensAt(active(lb2gt_0),138). orientation(lb2gt_3,159,138). coords(lb2gt_3,6,60,138).  visible(lb2gt_3,138). happensAt(walking(lb2gt_3),138).
orientation(lb2gt_0,141,139). coords(lb2gt_0,187,119,139).  visible(lb2gt_0,139). happensAt(active(lb2gt_0),139). orientation(lb2gt_3,161,139). coords(lb2gt_3,6,60,139).  visible(lb2gt_3,139). happensAt(walking(lb2gt_3),139).
orientation(lb2gt_0,141,140). coords(lb2gt_0,186,119,140).  visible(lb2gt_0,140). happensAt(active(lb2gt_0),140). orientation(lb2gt_3,166,140). coords(lb2gt_3,6,60,140).  visible(lb2gt_3,140). happensAt(walking(lb2gt_3),140).
orientation(lb2gt_0,141,141). coords(lb2gt_0,186,118,141).  visible(lb2gt_0,141). happensAt(active(lb2gt_0),141). orientation(lb2gt_3,160,141). coords(lb2gt_3,6,61,141).  visible(lb2gt_3,141). happensAt(walking(lb2gt_3),141).
orientation(lb2gt_0,141,142). coords(lb2gt_0,185,118,142).  visible(lb2gt_0,142). happensAt(active(lb2gt_0),142). orientation(lb2gt_3,161,142). coords(lb2gt_3,6,61,142).  visible(lb2gt_3,142). happensAt(walking(lb2gt_3),142).
orientation(lb2gt_0,141,143). coords(lb2gt_0,185,117,143).  visible(lb2gt_0,143). happensAt(active(lb2gt_0),143). orientation(lb2gt_3,160,143). coords(lb2gt_3,6,61,143).  visible(lb2gt_3,143). happensAt(walking(lb2gt_3),143).
orientation(lb2gt_0,141,144). coords(lb2gt_0,185,117,144).  visible(lb2gt_0,144). happensAt(active(lb2gt_0),144). orientation(lb2gt_3,159,144). coords(lb2gt_3,6,61,144).  visible(lb2gt_3,144). happensAt(walking(lb2gt_3),144).
orientation(lb2gt_0,141,145). coords(lb2gt_0,184,117,145).  visible(lb2gt_0,145). happensAt(active(lb2gt_0),145). orientation(lb2gt_3,163,145). coords(lb2gt_3,6,62,145).  visible(lb2gt_3,145). happensAt(walking(lb2gt_3),145).
orientation(lb2gt_0,141,146). coords(lb2gt_0,184,117,146).  visible(lb2gt_0,146). happensAt(active(lb2gt_0),146). orientation(lb2gt_3,168,146). coords(lb2gt_3,6,63,146).  visible(lb2gt_3,146). happensAt(walking(lb2gt_3),146).
orientation(lb2gt_0,141,147). coords(lb2gt_0,184,117,147).  visible(lb2gt_0,147). happensAt(active(lb2gt_0),147). orientation(lb2gt_3,168,147). coords(lb2gt_3,6,63,147).  visible(lb2gt_3,147). happensAt(walking(lb2gt_3),147).
orientation(lb2gt_0,141,148). coords(lb2gt_0,184,117,148).  visible(lb2gt_0,148). happensAt(active(lb2gt_0),148). orientation(lb2gt_3,168,148). coords(lb2gt_3,6,63,148).  visible(lb2gt_3,148). happensAt(walking(lb2gt_3),148).
orientation(lb2gt_0,141,149). coords(lb2gt_0,184,117,149).  visible(lb2gt_0,149). happensAt(active(lb2gt_0),149). orientation(lb2gt_3,172,149). coords(lb2gt_3,6,65,149).  visible(lb2gt_3,149). happensAt(walking(lb2gt_3),149).
orientation(lb2gt_0,141,150). coords(lb2gt_0,184,116,150).  visible(lb2gt_0,150). happensAt(active(lb2gt_0),150). orientation(lb2gt_3,170,150). coords(lb2gt_3,6,65,150).  visible(lb2gt_3,150). happensAt(walking(lb2gt_3),150).
orientation(lb2gt_0,141,151). coords(lb2gt_0,184,116,151).  visible(lb2gt_0,151). happensAt(active(lb2gt_0),151). orientation(lb2gt_3,172,151). coords(lb2gt_3,6,65,151).  visible(lb2gt_3,151). happensAt(walking(lb2gt_3),151).
orientation(lb2gt_0,141,152). coords(lb2gt_0,184,116,152).  visible(lb2gt_0,152). happensAt(active(lb2gt_0),152). orientation(lb2gt_3,173,152). coords(lb2gt_3,6,66,152).  visible(lb2gt_3,152). happensAt(walking(lb2gt_3),152).
orientation(lb2gt_0,141,153). coords(lb2gt_0,184,116,153).  visible(lb2gt_0,153). happensAt(active(lb2gt_0),153). orientation(lb2gt_3,175,153). coords(lb2gt_3,6,66,153).  visible(lb2gt_3,153). happensAt(walking(lb2gt_3),153).
orientation(lb2gt_0,141,154). coords(lb2gt_0,185,116,154).  visible(lb2gt_0,154). happensAt(active(lb2gt_0),154). orientation(lb2gt_3,172,154). coords(lb2gt_3,6,66,154).  visible(lb2gt_3,154). happensAt(walking(lb2gt_3),154).
orientation(lb2gt_0,141,155). coords(lb2gt_0,185,116,155).  visible(lb2gt_0,155). happensAt(active(lb2gt_0),155). orientation(lb2gt_3,173,155). coords(lb2gt_3,6,66,155).  visible(lb2gt_3,155). happensAt(walking(lb2gt_3),155).
orientation(lb2gt_0,141,156). coords(lb2gt_0,185,116,156).  visible(lb2gt_0,156). happensAt(active(lb2gt_0),156). orientation(lb2gt_3,171,156). coords(lb2gt_3,5,67,156).  visible(lb2gt_3,156). happensAt(walking(lb2gt_3),156).
#show holdsAt/2.