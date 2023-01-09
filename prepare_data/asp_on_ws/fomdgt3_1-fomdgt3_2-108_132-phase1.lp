
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

	orientation(fomdgt3_1,160,108). coords(fomdgt3_1,142,114,108).  visible(fomdgt3_1,108). happensAt(walking(fomdgt3_1),108). orientation(fomdgt3_2,90,108). coords(fomdgt3_2,283,282,108).  happensAt(appear(fomdgt3_2),108). happensAt(walking(fomdgt3_2),108).
orientation(fomdgt3_1,160,109). coords(fomdgt3_1,142,114,109).  visible(fomdgt3_1,109). happensAt(walking(fomdgt3_1),109). orientation(fomdgt3_2,90,109). coords(fomdgt3_2,283,282,109).  visible(fomdgt3_2,109). happensAt(walking(fomdgt3_2),109).
orientation(fomdgt3_1,160,110). coords(fomdgt3_1,141,114,110).  visible(fomdgt3_1,110). happensAt(walking(fomdgt3_1),110). orientation(fomdgt3_2,90,110). coords(fomdgt3_2,283,282,110).  visible(fomdgt3_2,110). happensAt(walking(fomdgt3_2),110).
orientation(fomdgt3_1,160,111). coords(fomdgt3_1,141,114,111).  visible(fomdgt3_1,111). happensAt(walking(fomdgt3_1),111). orientation(fomdgt3_2,90,111). coords(fomdgt3_2,283,282,111).  visible(fomdgt3_2,111). happensAt(walking(fomdgt3_2),111).
orientation(fomdgt3_1,160,112). coords(fomdgt3_1,141,114,112).  visible(fomdgt3_1,112). happensAt(walking(fomdgt3_1),112). orientation(fomdgt3_2,90,112). coords(fomdgt3_2,283,282,112).  visible(fomdgt3_2,112). happensAt(walking(fomdgt3_2),112).
orientation(fomdgt3_1,160,113). coords(fomdgt3_1,140,113,113).  visible(fomdgt3_1,113). happensAt(walking(fomdgt3_1),113). orientation(fomdgt3_2,90,113). coords(fomdgt3_2,283,282,113).  visible(fomdgt3_2,113). happensAt(walking(fomdgt3_2),113).
orientation(fomdgt3_1,150,114). coords(fomdgt3_1,140,113,114).  visible(fomdgt3_1,114). happensAt(walking(fomdgt3_1),114). orientation(fomdgt3_2,90,114). coords(fomdgt3_2,283,282,114).  visible(fomdgt3_2,114). happensAt(walking(fomdgt3_2),114).
orientation(fomdgt3_1,150,115). coords(fomdgt3_1,140,113,115).  visible(fomdgt3_1,115). happensAt(walking(fomdgt3_1),115). orientation(fomdgt3_2,90,115). coords(fomdgt3_2,283,282,115).  visible(fomdgt3_2,115). happensAt(walking(fomdgt3_2),115).
orientation(fomdgt3_1,150,116). coords(fomdgt3_1,138,111,116).  visible(fomdgt3_1,116). happensAt(walking(fomdgt3_1),116). orientation(fomdgt3_2,90,116). coords(fomdgt3_2,283,282,116).  visible(fomdgt3_2,116). happensAt(walking(fomdgt3_2),116).
orientation(fomdgt3_1,150,117). coords(fomdgt3_1,136,111,117).  visible(fomdgt3_1,117). happensAt(walking(fomdgt3_1),117). orientation(fomdgt3_2,90,117). coords(fomdgt3_2,283,282,117).  visible(fomdgt3_2,117). happensAt(walking(fomdgt3_2),117).
orientation(fomdgt3_1,150,118). coords(fomdgt3_1,136,111,118).  visible(fomdgt3_1,118). happensAt(walking(fomdgt3_1),118). orientation(fomdgt3_2,45,118). coords(fomdgt3_2,278,282,118).  visible(fomdgt3_2,118). happensAt(walking(fomdgt3_2),118).
orientation(fomdgt3_1,150,119). coords(fomdgt3_1,133,110,119).  visible(fomdgt3_1,119). happensAt(walking(fomdgt3_1),119). orientation(fomdgt3_2,40,119). coords(fomdgt3_2,278,279,119).  visible(fomdgt3_2,119). happensAt(walking(fomdgt3_2),119).
orientation(fomdgt3_1,150,120). coords(fomdgt3_1,130,110,120).  visible(fomdgt3_1,120). happensAt(walking(fomdgt3_1),120). orientation(fomdgt3_2,40,120). coords(fomdgt3_2,277,277,120).  visible(fomdgt3_2,120). happensAt(walking(fomdgt3_2),120).
orientation(fomdgt3_1,150,121). coords(fomdgt3_1,130,109,121).  visible(fomdgt3_1,121). happensAt(walking(fomdgt3_1),121). orientation(fomdgt3_2,40,121). coords(fomdgt3_2,275,272,121).  visible(fomdgt3_2,121). happensAt(walking(fomdgt3_2),121).
orientation(fomdgt3_1,150,122). coords(fomdgt3_1,130,109,122).  visible(fomdgt3_1,122). happensAt(walking(fomdgt3_1),122). orientation(fomdgt3_2,40,122). coords(fomdgt3_2,275,271,122).  visible(fomdgt3_2,122). happensAt(walking(fomdgt3_2),122).
orientation(fomdgt3_1,150,123). coords(fomdgt3_1,130,109,123).  visible(fomdgt3_1,123). happensAt(walking(fomdgt3_1),123). orientation(fomdgt3_2,40,123). coords(fomdgt3_2,275,271,123).  visible(fomdgt3_2,123). happensAt(walking(fomdgt3_2),123).
orientation(fomdgt3_1,150,124). coords(fomdgt3_1,130,109,124).  visible(fomdgt3_1,124). happensAt(walking(fomdgt3_1),124). orientation(fomdgt3_2,40,124). coords(fomdgt3_2,275,271,124).  visible(fomdgt3_2,124). happensAt(walking(fomdgt3_2),124).
orientation(fomdgt3_1,150,125). coords(fomdgt3_1,129,109,125).  visible(fomdgt3_1,125). happensAt(walking(fomdgt3_1),125). orientation(fomdgt3_2,50,125). coords(fomdgt3_2,275,269,125).  visible(fomdgt3_2,125). happensAt(walking(fomdgt3_2),125).
orientation(fomdgt3_1,150,126). coords(fomdgt3_1,129,109,126).  visible(fomdgt3_1,126). happensAt(walking(fomdgt3_1),126). orientation(fomdgt3_2,50,126). coords(fomdgt3_2,275,269,126).  visible(fomdgt3_2,126). happensAt(walking(fomdgt3_2),126).
orientation(fomdgt3_1,150,127). coords(fomdgt3_1,129,109,127).  visible(fomdgt3_1,127). happensAt(walking(fomdgt3_1),127). orientation(fomdgt3_2,50,127). coords(fomdgt3_2,275,269,127).  visible(fomdgt3_2,127). happensAt(walking(fomdgt3_2),127).
orientation(fomdgt3_1,150,128). coords(fomdgt3_1,126,108,128).  visible(fomdgt3_1,128). happensAt(walking(fomdgt3_1),128). orientation(fomdgt3_2,50,128). coords(fomdgt3_2,275,267,128).  visible(fomdgt3_2,128). happensAt(walking(fomdgt3_2),128).
orientation(fomdgt3_1,150,129). coords(fomdgt3_1,126,108,129).  visible(fomdgt3_1,129). happensAt(walking(fomdgt3_1),129). orientation(fomdgt3_2,50,129). coords(fomdgt3_2,275,267,129).  visible(fomdgt3_2,129). happensAt(walking(fomdgt3_2),129).
orientation(fomdgt3_1,150,130). coords(fomdgt3_1,124,108,130).  visible(fomdgt3_1,130). happensAt(walking(fomdgt3_1),130). orientation(fomdgt3_2,50,130). coords(fomdgt3_2,275,265,130).  visible(fomdgt3_2,130). happensAt(walking(fomdgt3_2),130).
orientation(fomdgt3_1,150,131). coords(fomdgt3_1,121,108,131).  visible(fomdgt3_1,131). happensAt(walking(fomdgt3_1),131). orientation(fomdgt3_2,50,131). coords(fomdgt3_2,275,263,131).  visible(fomdgt3_2,131). happensAt(walking(fomdgt3_2),131).
orientation(fomdgt3_1,150,132). coords(fomdgt3_1,121,108,132).  visible(fomdgt3_1,132). happensAt(walking(fomdgt3_1),132). orientation(fomdgt3_2,50,132). coords(fomdgt3_2,275,262,132).  visible(fomdgt3_2,132). happensAt(walking(fomdgt3_2),132).
#show holdsAt/2.