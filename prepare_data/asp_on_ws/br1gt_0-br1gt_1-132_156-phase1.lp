
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

	orientation(br1gt_0,152,132). coords(br1gt_0,213,174,132).  visible(br1gt_0,132). happensAt(walking(br1gt_0),132). orientation(br1gt_1,164,132). coords(br1gt_1,56,161,132).  visible(br1gt_1,132). happensAt(walking(br1gt_1),132).
orientation(br1gt_0,152,133). coords(br1gt_0,216,175,133).  visible(br1gt_0,133). happensAt(walking(br1gt_0),133). orientation(br1gt_1,164,133). coords(br1gt_1,55,161,133).  visible(br1gt_1,133). happensAt(walking(br1gt_1),133).
orientation(br1gt_0,152,134). coords(br1gt_0,217,177,134).  visible(br1gt_0,134). happensAt(walking(br1gt_0),134). orientation(br1gt_1,164,134). coords(br1gt_1,55,161,134).  visible(br1gt_1,134). happensAt(walking(br1gt_1),134).
orientation(br1gt_0,152,135). coords(br1gt_0,218,178,135).  visible(br1gt_0,135). happensAt(walking(br1gt_0),135). orientation(br1gt_1,164,135). coords(br1gt_1,53,161,135).  visible(br1gt_1,135). happensAt(walking(br1gt_1),135).
orientation(br1gt_0,152,136). coords(br1gt_0,219,179,136).  visible(br1gt_0,136). happensAt(walking(br1gt_0),136). orientation(br1gt_1,164,136). coords(br1gt_1,53,161,136).  visible(br1gt_1,136). happensAt(walking(br1gt_1),136).
orientation(br1gt_0,152,137). coords(br1gt_0,220,179,137).  visible(br1gt_0,137). happensAt(walking(br1gt_0),137). orientation(br1gt_1,164,137). coords(br1gt_1,53,161,137).  visible(br1gt_1,137). happensAt(walking(br1gt_1),137).
orientation(br1gt_0,154,138). coords(br1gt_0,221,180,138).  visible(br1gt_0,138). happensAt(walking(br1gt_0),138). orientation(br1gt_1,164,138). coords(br1gt_1,53,161,138).  visible(br1gt_1,138). happensAt(walking(br1gt_1),138).
orientation(br1gt_0,154,139). coords(br1gt_0,222,181,139).  visible(br1gt_0,139). happensAt(walking(br1gt_0),139). orientation(br1gt_1,164,139). coords(br1gt_1,53,161,139).  visible(br1gt_1,139). happensAt(walking(br1gt_1),139).
orientation(br1gt_0,156,140). coords(br1gt_0,222,182,140).  visible(br1gt_0,140). happensAt(walking(br1gt_0),140). orientation(br1gt_1,164,140). coords(br1gt_1,53,161,140).  visible(br1gt_1,140). happensAt(walking(br1gt_1),140).
orientation(br1gt_0,159,141). coords(br1gt_0,226,183,141).  visible(br1gt_0,141). happensAt(walking(br1gt_0),141). orientation(br1gt_1,164,141). coords(br1gt_1,52,160,141).  visible(br1gt_1,141). happensAt(walking(br1gt_1),141).
orientation(br1gt_0,165,142). coords(br1gt_0,229,184,142).  visible(br1gt_0,142). happensAt(walking(br1gt_0),142). orientation(br1gt_1,164,142). coords(br1gt_1,52,160,142).  visible(br1gt_1,142). happensAt(walking(br1gt_1),142).
orientation(br1gt_0,165,143). coords(br1gt_0,232,185,143).  visible(br1gt_0,143). happensAt(walking(br1gt_0),143). orientation(br1gt_1,164,143). coords(br1gt_1,51,160,143).  visible(br1gt_1,143). happensAt(walking(br1gt_1),143).
orientation(br1gt_0,159,144). coords(br1gt_0,235,189,144).  visible(br1gt_0,144). happensAt(walking(br1gt_0),144). orientation(br1gt_1,164,144). coords(br1gt_1,51,160,144).  visible(br1gt_1,144). happensAt(walking(br1gt_1),144).
orientation(br1gt_0,159,145). coords(br1gt_0,238,192,145).  visible(br1gt_0,145). happensAt(walking(br1gt_0),145). orientation(br1gt_1,164,145). coords(br1gt_1,50,159,145).  visible(br1gt_1,145). happensAt(walking(br1gt_1),145).
orientation(br1gt_0,159,146). coords(br1gt_0,241,194,146).  visible(br1gt_0,146). happensAt(walking(br1gt_0),146). orientation(br1gt_1,164,146). coords(br1gt_1,50,159,146).  visible(br1gt_1,146). happensAt(walking(br1gt_1),146).
orientation(br1gt_0,159,147). coords(br1gt_0,242,197,147).  visible(br1gt_0,147). happensAt(walking(br1gt_0),147). orientation(br1gt_1,164,147). coords(br1gt_1,49,158,147).  visible(br1gt_1,147). happensAt(walking(br1gt_1),147).
orientation(br1gt_0,161,148). coords(br1gt_0,244,198,148).  visible(br1gt_0,148). happensAt(walking(br1gt_0),148). orientation(br1gt_1,164,148). coords(br1gt_1,49,158,148).  visible(br1gt_1,148). happensAt(walking(br1gt_1),148).
orientation(br1gt_0,155,149). coords(br1gt_0,244,200,149).  visible(br1gt_0,149). happensAt(walking(br1gt_0),149). orientation(br1gt_1,164,149). coords(br1gt_1,49,158,149).  visible(br1gt_1,149). happensAt(walking(br1gt_1),149).
orientation(br1gt_0,160,150). coords(br1gt_0,245,201,150).  visible(br1gt_0,150). happensAt(walking(br1gt_0),150). orientation(br1gt_1,164,150). coords(br1gt_1,49,158,150).  visible(br1gt_1,150). happensAt(walking(br1gt_1),150).
orientation(br1gt_0,161,151). coords(br1gt_0,246,204,151).  visible(br1gt_0,151). happensAt(walking(br1gt_0),151). orientation(br1gt_1,164,151). coords(br1gt_1,49,158,151).  visible(br1gt_1,151). happensAt(walking(br1gt_1),151).
orientation(br1gt_0,167,152). coords(br1gt_0,247,206,152).  visible(br1gt_0,152). happensAt(walking(br1gt_0),152). orientation(br1gt_1,164,152). coords(br1gt_1,49,157,152).  visible(br1gt_1,152). happensAt(walking(br1gt_1),152).
orientation(br1gt_0,160,153). coords(br1gt_0,248,208,153).  visible(br1gt_0,153). happensAt(walking(br1gt_0),153). orientation(br1gt_1,164,153). coords(br1gt_1,49,157,153).  visible(br1gt_1,153). happensAt(walking(br1gt_1),153).
orientation(br1gt_0,160,154). coords(br1gt_0,249,210,154).  visible(br1gt_0,154). happensAt(walking(br1gt_0),154). orientation(br1gt_1,164,154). coords(br1gt_1,49,157,154).  visible(br1gt_1,154). happensAt(walking(br1gt_1),154).
orientation(br1gt_0,160,155). coords(br1gt_0,251,213,155).  visible(br1gt_0,155). happensAt(walking(br1gt_0),155). orientation(br1gt_1,164,155). coords(br1gt_1,48,156,155).  visible(br1gt_1,155). happensAt(walking(br1gt_1),155).
orientation(br1gt_0,160,156). coords(br1gt_0,252,216,156).  visible(br1gt_0,156). happensAt(walking(br1gt_0),156). orientation(br1gt_1,164,156). coords(br1gt_1,47,156,156).  visible(br1gt_1,156). happensAt(walking(br1gt_1),156).
#show holdsAt/2.