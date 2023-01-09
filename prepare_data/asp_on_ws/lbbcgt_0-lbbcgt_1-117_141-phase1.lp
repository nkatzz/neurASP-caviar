
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

	orientation(lbbcgt_0,7,117). coords(lbbcgt_0,201,184,117).  visible(lbbcgt_0,117). happensAt(walking(lbbcgt_0),117). orientation(lbbcgt_1,147,117). coords(lbbcgt_1,16,45,117).  happensAt(appear(lbbcgt_1),117). happensAt(walking(lbbcgt_1),117).
orientation(lbbcgt_0,11,118). coords(lbbcgt_0,201,187,118).  visible(lbbcgt_0,118). happensAt(walking(lbbcgt_0),118). orientation(lbbcgt_1,147,118). coords(lbbcgt_1,16,46,118).  visible(lbbcgt_1,118). happensAt(walking(lbbcgt_1),118).
orientation(lbbcgt_0,11,119). coords(lbbcgt_0,203,190,119).  visible(lbbcgt_0,119). happensAt(walking(lbbcgt_0),119). orientation(lbbcgt_1,147,119). coords(lbbcgt_1,16,46,119).  visible(lbbcgt_1,119). happensAt(walking(lbbcgt_1),119).
orientation(lbbcgt_0,173,120). coords(lbbcgt_0,203,194,120).  visible(lbbcgt_0,120). happensAt(walking(lbbcgt_0),120). orientation(lbbcgt_1,147,120). coords(lbbcgt_1,16,46,120).  visible(lbbcgt_1,120). happensAt(walking(lbbcgt_1),120).
orientation(lbbcgt_0,168,121). coords(lbbcgt_0,203,198,121).  visible(lbbcgt_0,121). happensAt(walking(lbbcgt_0),121). orientation(lbbcgt_1,147,121). coords(lbbcgt_1,16,46,121).  visible(lbbcgt_1,121). happensAt(walking(lbbcgt_1),121).
orientation(lbbcgt_0,169,122). coords(lbbcgt_0,203,201,122).  visible(lbbcgt_0,122). happensAt(walking(lbbcgt_0),122). orientation(lbbcgt_1,147,122). coords(lbbcgt_1,16,47,122).  visible(lbbcgt_1,122). happensAt(walking(lbbcgt_1),122).
orientation(lbbcgt_0,168,123). coords(lbbcgt_0,205,204,123).  visible(lbbcgt_0,123). happensAt(walking(lbbcgt_0),123). orientation(lbbcgt_1,147,123). coords(lbbcgt_1,16,47,123).  happensAt(occluded(lbbcgt_1),123). happensAt(walking(lbbcgt_1),123).
orientation(lbbcgt_0,171,124). coords(lbbcgt_0,206,208,124).  visible(lbbcgt_0,124). happensAt(walking(lbbcgt_0),124). orientation(lbbcgt_1,147,124). coords(lbbcgt_1,16,47,124).  happensAt(occluded(lbbcgt_1),124). happensAt(walking(lbbcgt_1),124).
orientation(lbbcgt_0,173,125). coords(lbbcgt_0,208,211,125).  visible(lbbcgt_0,125). happensAt(walking(lbbcgt_0),125). orientation(lbbcgt_1,147,125). coords(lbbcgt_1,16,47,125).  happensAt(occluded(lbbcgt_1),125). happensAt(walking(lbbcgt_1),125).
orientation(lbbcgt_0,173,126). coords(lbbcgt_0,209,213,126).  visible(lbbcgt_0,126). happensAt(walking(lbbcgt_0),126). orientation(lbbcgt_1,147,126). coords(lbbcgt_1,16,47,126).  happensAt(occluded(lbbcgt_1),126). happensAt(walking(lbbcgt_1),126).
orientation(lbbcgt_0,173,127). coords(lbbcgt_0,211,214,127).  visible(lbbcgt_0,127). happensAt(walking(lbbcgt_0),127). orientation(lbbcgt_1,150,127). coords(lbbcgt_1,16,55,127).  visible(lbbcgt_1,127). happensAt(walking(lbbcgt_1),127).
orientation(lbbcgt_0,12,128). coords(lbbcgt_0,212,216,128).  visible(lbbcgt_0,128). happensAt(walking(lbbcgt_0),128). orientation(lbbcgt_1,148,128). coords(lbbcgt_1,11,53,128).  visible(lbbcgt_1,128). happensAt(walking(lbbcgt_1),128).
orientation(lbbcgt_0,12,129). coords(lbbcgt_0,213,219,129).  visible(lbbcgt_0,129). happensAt(walking(lbbcgt_0),129). orientation(lbbcgt_1,150,129). coords(lbbcgt_1,11,53,129).  visible(lbbcgt_1,129). happensAt(walking(lbbcgt_1),129).
orientation(lbbcgt_0,12,130). coords(lbbcgt_0,214,222,130).  visible(lbbcgt_0,130). happensAt(walking(lbbcgt_0),130). orientation(lbbcgt_1,148,130). coords(lbbcgt_1,13,53,130).  visible(lbbcgt_1,130). happensAt(walking(lbbcgt_1),130).
orientation(lbbcgt_0,12,131). coords(lbbcgt_0,216,228,131).  visible(lbbcgt_0,131). happensAt(walking(lbbcgt_0),131). orientation(lbbcgt_1,150,131). coords(lbbcgt_1,13,54,131).  visible(lbbcgt_1,131). happensAt(walking(lbbcgt_1),131).
orientation(lbbcgt_0,12,132). coords(lbbcgt_0,219,232,132).  visible(lbbcgt_0,132). happensAt(walking(lbbcgt_0),132). orientation(lbbcgt_1,152,132). coords(lbbcgt_1,13,55,132).  visible(lbbcgt_1,132). happensAt(walking(lbbcgt_1),132).
orientation(lbbcgt_0,12,133). coords(lbbcgt_0,222,235,133).  visible(lbbcgt_0,133). happensAt(walking(lbbcgt_0),133). orientation(lbbcgt_1,147,133). coords(lbbcgt_1,13,56,133).  visible(lbbcgt_1,133). happensAt(walking(lbbcgt_1),133).
orientation(lbbcgt_0,12,134). coords(lbbcgt_0,225,238,134).  visible(lbbcgt_0,134). happensAt(walking(lbbcgt_0),134). orientation(lbbcgt_1,147,134). coords(lbbcgt_1,13,56,134).  visible(lbbcgt_1,134). happensAt(walking(lbbcgt_1),134).
orientation(lbbcgt_0,22,135). coords(lbbcgt_0,227,240,135).  visible(lbbcgt_0,135). happensAt(walking(lbbcgt_0),135). orientation(lbbcgt_1,147,135). coords(lbbcgt_1,13,56,135).  visible(lbbcgt_1,135). happensAt(walking(lbbcgt_1),135).
orientation(lbbcgt_0,22,136). coords(lbbcgt_0,229,242,136).  visible(lbbcgt_0,136). happensAt(walking(lbbcgt_0),136). orientation(lbbcgt_1,147,136). coords(lbbcgt_1,13,56,136).  visible(lbbcgt_1,136). happensAt(walking(lbbcgt_1),136).
orientation(lbbcgt_0,22,137). coords(lbbcgt_0,230,245,137).  visible(lbbcgt_0,137). happensAt(walking(lbbcgt_0),137). orientation(lbbcgt_1,147,137). coords(lbbcgt_1,13,57,137).  visible(lbbcgt_1,137). happensAt(walking(lbbcgt_1),137).
orientation(lbbcgt_0,22,138). coords(lbbcgt_0,232,247,138).  visible(lbbcgt_0,138). happensAt(walking(lbbcgt_0),138). orientation(lbbcgt_1,147,138). coords(lbbcgt_1,13,58,138).  visible(lbbcgt_1,138). happensAt(walking(lbbcgt_1),138).
orientation(lbbcgt_0,22,139). coords(lbbcgt_0,233,249,139).  visible(lbbcgt_0,139). happensAt(walking(lbbcgt_0),139). orientation(lbbcgt_1,147,139). coords(lbbcgt_1,12,59,139).  visible(lbbcgt_1,139). happensAt(walking(lbbcgt_1),139).
orientation(lbbcgt_0,22,140). coords(lbbcgt_0,234,252,140).  visible(lbbcgt_0,140). happensAt(walking(lbbcgt_0),140). orientation(lbbcgt_1,147,140). coords(lbbcgt_1,12,60,140).  visible(lbbcgt_1,140). happensAt(walking(lbbcgt_1),140).
orientation(lbbcgt_0,22,141). coords(lbbcgt_0,235,256,141).  visible(lbbcgt_0,141). happensAt(walking(lbbcgt_0),141). orientation(lbbcgt_1,147,141). coords(lbbcgt_1,12,60,141).  visible(lbbcgt_1,141). happensAt(walking(lbbcgt_1),141).
#show holdsAt/2.