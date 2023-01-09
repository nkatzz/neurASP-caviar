
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

	orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). happensAt(walking(mc1gt_0),124). orientation(mc1gt_1,145,124). coords(mc1gt_1,107,82,124).  visible(mc1gt_1,124). happensAt(walking(mc1gt_1),124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). happensAt(walking(mc1gt_0),125). orientation(mc1gt_1,145,125). coords(mc1gt_1,108,82,125).  visible(mc1gt_1,125). happensAt(walking(mc1gt_1),125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). happensAt(walking(mc1gt_0),126). orientation(mc1gt_1,145,126). coords(mc1gt_1,109,82,126).  visible(mc1gt_1,126). happensAt(walking(mc1gt_1),126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). happensAt(walking(mc1gt_0),127). orientation(mc1gt_1,145,127). coords(mc1gt_1,110,82,127).  visible(mc1gt_1,127). happensAt(walking(mc1gt_1),127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). happensAt(walking(mc1gt_0),128). orientation(mc1gt_1,145,128). coords(mc1gt_1,110,82,128).  visible(mc1gt_1,128). happensAt(walking(mc1gt_1),128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). happensAt(walking(mc1gt_0),129). orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129). happensAt(walking(mc1gt_1),129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). happensAt(walking(mc1gt_0),130). orientation(mc1gt_1,145,130). coords(mc1gt_1,111,82,130).  visible(mc1gt_1,130). happensAt(walking(mc1gt_1),130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). happensAt(walking(mc1gt_0),131). orientation(mc1gt_1,145,131). coords(mc1gt_1,111,81,131).  visible(mc1gt_1,131). happensAt(walking(mc1gt_1),131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). happensAt(walking(mc1gt_0),132). orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132). happensAt(walking(mc1gt_1),132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). happensAt(walking(mc1gt_0),133). orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133). happensAt(walking(mc1gt_1),133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). happensAt(walking(mc1gt_0),134). orientation(mc1gt_1,145,134). coords(mc1gt_1,112,81,134).  visible(mc1gt_1,134). happensAt(walking(mc1gt_1),134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). happensAt(walking(mc1gt_0),135). orientation(mc1gt_1,145,135). coords(mc1gt_1,113,81,135).  visible(mc1gt_1,135). happensAt(walking(mc1gt_1),135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). happensAt(walking(mc1gt_0),136). orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136). happensAt(walking(mc1gt_1),136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). happensAt(walking(mc1gt_0),137). orientation(mc1gt_1,145,137). coords(mc1gt_1,113,81,137).  visible(mc1gt_1,137). happensAt(walking(mc1gt_1),137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). happensAt(walking(mc1gt_0),138). orientation(mc1gt_1,145,138). coords(mc1gt_1,114,82,138).  visible(mc1gt_1,138). happensAt(walking(mc1gt_1),138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). happensAt(walking(mc1gt_0),139). orientation(mc1gt_1,145,139). coords(mc1gt_1,116,82,139).  visible(mc1gt_1,139). happensAt(walking(mc1gt_1),139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). happensAt(walking(mc1gt_0),140). orientation(mc1gt_1,145,140). coords(mc1gt_1,116,82,140).  visible(mc1gt_1,140). happensAt(walking(mc1gt_1),140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). happensAt(walking(mc1gt_0),141). orientation(mc1gt_1,145,141). coords(mc1gt_1,117,82,141).  visible(mc1gt_1,141). happensAt(walking(mc1gt_1),141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). happensAt(walking(mc1gt_0),142). orientation(mc1gt_1,145,142). coords(mc1gt_1,118,83,142).  visible(mc1gt_1,142). happensAt(walking(mc1gt_1),142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). happensAt(walking(mc1gt_0),143). orientation(mc1gt_1,145,143). coords(mc1gt_1,118,83,143).  visible(mc1gt_1,143). happensAt(walking(mc1gt_1),143).
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). happensAt(walking(mc1gt_0),144). orientation(mc1gt_1,145,144). coords(mc1gt_1,119,83,144).  visible(mc1gt_1,144). happensAt(walking(mc1gt_1),144).
orientation(mc1gt_0,147,145). coords(mc1gt_0,95,107,145).  visible(mc1gt_0,145). happensAt(walking(mc1gt_0),145). orientation(mc1gt_1,145,145). coords(mc1gt_1,119,83,145).  visible(mc1gt_1,145). happensAt(walking(mc1gt_1),145).
orientation(mc1gt_0,147,146). coords(mc1gt_0,96,107,146).  visible(mc1gt_0,146). happensAt(walking(mc1gt_0),146). orientation(mc1gt_1,145,146). coords(mc1gt_1,119,83,146).  visible(mc1gt_1,146). happensAt(walking(mc1gt_1),146).
orientation(mc1gt_0,147,147). coords(mc1gt_0,98,106,147).  visible(mc1gt_0,147). happensAt(walking(mc1gt_0),147). orientation(mc1gt_1,145,147). coords(mc1gt_1,120,83,147).  visible(mc1gt_1,147). happensAt(walking(mc1gt_1),147).
orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). happensAt(walking(mc1gt_0),148). orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148). happensAt(walking(mc1gt_1),148).
#show holdsAt/2.