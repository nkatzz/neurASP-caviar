
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

	orientation(mc1gt_0,147,117). coords(mc1gt_0,82,101,117).  visible(mc1gt_0,117). happensAt(walking(mc1gt_0),117). orientation(mc1gt_2,144,117). coords(mc1gt_2,77,78,117).  visible(mc1gt_2,117). happensAt(walking(mc1gt_2),117).
orientation(mc1gt_0,147,118). coords(mc1gt_0,83,102,118).  visible(mc1gt_0,118). happensAt(walking(mc1gt_0),118). orientation(mc1gt_2,144,118). coords(mc1gt_2,77,79,118).  visible(mc1gt_2,118). happensAt(walking(mc1gt_2),118).
orientation(mc1gt_0,147,119). coords(mc1gt_0,83,102,119).  visible(mc1gt_0,119). happensAt(walking(mc1gt_0),119). orientation(mc1gt_2,144,119). coords(mc1gt_2,77,79,119).  visible(mc1gt_2,119). happensAt(walking(mc1gt_2),119).
orientation(mc1gt_0,147,120). coords(mc1gt_0,84,102,120).  visible(mc1gt_0,120). happensAt(walking(mc1gt_0),120). orientation(mc1gt_2,144,120). coords(mc1gt_2,77,79,120).  visible(mc1gt_2,120). happensAt(walking(mc1gt_2),120).
orientation(mc1gt_0,147,121). coords(mc1gt_0,85,102,121).  visible(mc1gt_0,121). happensAt(walking(mc1gt_0),121). orientation(mc1gt_2,144,121). coords(mc1gt_2,77,80,121).  visible(mc1gt_2,121). happensAt(walking(mc1gt_2),121).
orientation(mc1gt_0,147,122). coords(mc1gt_0,86,102,122).  visible(mc1gt_0,122). happensAt(walking(mc1gt_0),122). orientation(mc1gt_2,144,122). coords(mc1gt_2,77,80,122).  visible(mc1gt_2,122). happensAt(walking(mc1gt_2),122).
orientation(mc1gt_0,147,123). coords(mc1gt_0,87,102,123).  visible(mc1gt_0,123). happensAt(walking(mc1gt_0),123). orientation(mc1gt_2,144,123). coords(mc1gt_2,77,81,123).  visible(mc1gt_2,123). happensAt(walking(mc1gt_2),123).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). happensAt(walking(mc1gt_0),124). orientation(mc1gt_2,144,124). coords(mc1gt_2,77,83,124).  visible(mc1gt_2,124). happensAt(walking(mc1gt_2),124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). happensAt(walking(mc1gt_0),125). orientation(mc1gt_2,144,125). coords(mc1gt_2,76,85,125).  visible(mc1gt_2,125). happensAt(walking(mc1gt_2),125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). happensAt(walking(mc1gt_0),126). orientation(mc1gt_2,144,126). coords(mc1gt_2,76,86,126).  visible(mc1gt_2,126). happensAt(walking(mc1gt_2),126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). happensAt(walking(mc1gt_0),127). orientation(mc1gt_2,144,127). coords(mc1gt_2,76,86,127).  visible(mc1gt_2,127). happensAt(walking(mc1gt_2),127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). happensAt(walking(mc1gt_0),128). orientation(mc1gt_2,144,128). coords(mc1gt_2,76,87,128).  visible(mc1gt_2,128). happensAt(walking(mc1gt_2),128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). happensAt(walking(mc1gt_0),129). orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129). happensAt(walking(mc1gt_2),129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). happensAt(walking(mc1gt_0),130). orientation(mc1gt_2,144,130). coords(mc1gt_2,76,88,130).  visible(mc1gt_2,130). happensAt(walking(mc1gt_2),130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). happensAt(walking(mc1gt_0),131). orientation(mc1gt_2,144,131). coords(mc1gt_2,76,88,131).  visible(mc1gt_2,131). happensAt(walking(mc1gt_2),131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). happensAt(walking(mc1gt_0),132). orientation(mc1gt_2,144,132). coords(mc1gt_2,76,88,132).  visible(mc1gt_2,132). happensAt(walking(mc1gt_2),132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). happensAt(walking(mc1gt_0),133). orientation(mc1gt_2,144,133). coords(mc1gt_2,76,89,133).  visible(mc1gt_2,133). happensAt(walking(mc1gt_2),133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). happensAt(walking(mc1gt_0),134). orientation(mc1gt_2,144,134). coords(mc1gt_2,76,89,134).  visible(mc1gt_2,134). happensAt(walking(mc1gt_2),134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). happensAt(walking(mc1gt_0),135). orientation(mc1gt_2,144,135). coords(mc1gt_2,76,90,135).  visible(mc1gt_2,135). happensAt(walking(mc1gt_2),135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). happensAt(walking(mc1gt_0),136). orientation(mc1gt_2,144,136). coords(mc1gt_2,76,90,136).  visible(mc1gt_2,136). happensAt(walking(mc1gt_2),136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). happensAt(walking(mc1gt_0),137). orientation(mc1gt_2,144,137). coords(mc1gt_2,77,90,137).  visible(mc1gt_2,137). happensAt(walking(mc1gt_2),137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). happensAt(walking(mc1gt_0),138). orientation(mc1gt_2,144,138). coords(mc1gt_2,77,91,138).  visible(mc1gt_2,138). happensAt(walking(mc1gt_2),138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). happensAt(walking(mc1gt_0),139). orientation(mc1gt_2,144,139). coords(mc1gt_2,77,92,139).  visible(mc1gt_2,139). happensAt(walking(mc1gt_2),139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). happensAt(walking(mc1gt_0),140). orientation(mc1gt_2,144,140). coords(mc1gt_2,76,92,140).  visible(mc1gt_2,140). happensAt(walking(mc1gt_2),140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). happensAt(walking(mc1gt_0),141). orientation(mc1gt_2,144,141). coords(mc1gt_2,75,93,141).  visible(mc1gt_2,141). happensAt(walking(mc1gt_2),141).
#show holdsAt/2.