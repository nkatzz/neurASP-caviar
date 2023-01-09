
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

	orientation(mc1gt_0,147,120). coords(mc1gt_0,84,102,120).  visible(mc1gt_0,120). happensAt(walking(mc1gt_0),120). orientation(mc1gt_3,146,120). coords(mc1gt_3,80,71,120).  visible(mc1gt_3,120). happensAt(walking(mc1gt_3),120).
orientation(mc1gt_0,147,121). coords(mc1gt_0,85,102,121).  visible(mc1gt_0,121). happensAt(walking(mc1gt_0),121). orientation(mc1gt_3,146,121). coords(mc1gt_3,81,71,121).  visible(mc1gt_3,121). happensAt(walking(mc1gt_3),121).
orientation(mc1gt_0,147,122). coords(mc1gt_0,86,102,122).  visible(mc1gt_0,122). happensAt(walking(mc1gt_0),122). orientation(mc1gt_3,146,122). coords(mc1gt_3,83,73,122).  visible(mc1gt_3,122). happensAt(walking(mc1gt_3),122).
orientation(mc1gt_0,147,123). coords(mc1gt_0,87,102,123).  visible(mc1gt_0,123). happensAt(walking(mc1gt_0),123). orientation(mc1gt_3,146,123). coords(mc1gt_3,83,73,123).  visible(mc1gt_3,123). happensAt(walking(mc1gt_3),123).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). happensAt(walking(mc1gt_0),124). orientation(mc1gt_3,146,124). coords(mc1gt_3,83,73,124).  visible(mc1gt_3,124). happensAt(walking(mc1gt_3),124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). happensAt(walking(mc1gt_0),125). orientation(mc1gt_3,146,125). coords(mc1gt_3,83,73,125).  visible(mc1gt_3,125). happensAt(walking(mc1gt_3),125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). happensAt(walking(mc1gt_0),126). orientation(mc1gt_3,146,126). coords(mc1gt_3,84,74,126).  visible(mc1gt_3,126). happensAt(walking(mc1gt_3),126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). happensAt(walking(mc1gt_0),127). orientation(mc1gt_3,146,127). coords(mc1gt_3,84,74,127).  visible(mc1gt_3,127). happensAt(walking(mc1gt_3),127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). happensAt(walking(mc1gt_0),128). orientation(mc1gt_3,146,128). coords(mc1gt_3,84,74,128).  visible(mc1gt_3,128). happensAt(walking(mc1gt_3),128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). happensAt(walking(mc1gt_0),129). orientation(mc1gt_3,146,129). coords(mc1gt_3,84,75,129).  visible(mc1gt_3,129). happensAt(walking(mc1gt_3),129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). happensAt(walking(mc1gt_0),130). orientation(mc1gt_3,146,130). coords(mc1gt_3,85,76,130).  visible(mc1gt_3,130). happensAt(walking(mc1gt_3),130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). happensAt(walking(mc1gt_0),131). orientation(mc1gt_3,146,131). coords(mc1gt_3,85,77,131).  visible(mc1gt_3,131). happensAt(walking(mc1gt_3),131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). happensAt(walking(mc1gt_0),132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132). happensAt(walking(mc1gt_3),132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). happensAt(walking(mc1gt_0),133). orientation(mc1gt_3,146,133). coords(mc1gt_3,87,79,133).  visible(mc1gt_3,133). happensAt(walking(mc1gt_3),133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). happensAt(walking(mc1gt_0),134). orientation(mc1gt_3,146,134). coords(mc1gt_3,89,79,134).  visible(mc1gt_3,134). happensAt(walking(mc1gt_3),134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). happensAt(walking(mc1gt_0),135). orientation(mc1gt_3,146,135). coords(mc1gt_3,89,78,135).  visible(mc1gt_3,135). happensAt(walking(mc1gt_3),135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). happensAt(walking(mc1gt_0),136). orientation(mc1gt_3,146,136). coords(mc1gt_3,89,78,136).  visible(mc1gt_3,136). happensAt(walking(mc1gt_3),136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). happensAt(walking(mc1gt_0),137). orientation(mc1gt_3,146,137). coords(mc1gt_3,90,78,137).  visible(mc1gt_3,137). happensAt(walking(mc1gt_3),137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). happensAt(walking(mc1gt_0),138). orientation(mc1gt_3,146,138). coords(mc1gt_3,91,79,138).  visible(mc1gt_3,138). happensAt(walking(mc1gt_3),138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). happensAt(walking(mc1gt_0),139). orientation(mc1gt_3,146,139). coords(mc1gt_3,91,79,139).  visible(mc1gt_3,139). happensAt(walking(mc1gt_3),139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). happensAt(walking(mc1gt_0),140). orientation(mc1gt_3,146,140). coords(mc1gt_3,91,79,140).  visible(mc1gt_3,140). happensAt(walking(mc1gt_3),140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). happensAt(walking(mc1gt_0),141). orientation(mc1gt_3,146,141). coords(mc1gt_3,91,79,141).  visible(mc1gt_3,141). happensAt(walking(mc1gt_3),141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). happensAt(walking(mc1gt_0),142). orientation(mc1gt_3,146,142). coords(mc1gt_3,91,79,142).  visible(mc1gt_3,142). happensAt(walking(mc1gt_3),142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). happensAt(walking(mc1gt_0),143). orientation(mc1gt_3,146,143). coords(mc1gt_3,91,79,143).  visible(mc1gt_3,143). happensAt(walking(mc1gt_3),143).
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). happensAt(walking(mc1gt_0),144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144). happensAt(walking(mc1gt_3),144).
#show holdsAt/2.