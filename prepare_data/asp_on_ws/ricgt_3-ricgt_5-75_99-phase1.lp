
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

	orientation(ricgt_3,10,75). coords(ricgt_3,246,242,75).  visible(ricgt_3,75). happensAt(walking(ricgt_3),75). orientation(ricgt_5,0,75). coords(ricgt_5,28,124,75).  visible(ricgt_5,75). happensAt(walking(ricgt_5),75).
orientation(ricgt_3,14,76). coords(ricgt_3,244,236,76).  visible(ricgt_3,76). happensAt(walking(ricgt_3),76). orientation(ricgt_5,0,76). coords(ricgt_5,28,124,76).  visible(ricgt_5,76). happensAt(walking(ricgt_5),76).
orientation(ricgt_3,12,77). coords(ricgt_3,242,233,77).  visible(ricgt_3,77). happensAt(walking(ricgt_3),77). orientation(ricgt_5,0,77). coords(ricgt_5,28,124,77).  visible(ricgt_5,77). happensAt(walking(ricgt_5),77).
orientation(ricgt_3,14,78). coords(ricgt_3,240,231,78).  visible(ricgt_3,78). happensAt(walking(ricgt_3),78). orientation(ricgt_5,0,78). coords(ricgt_5,28,124,78).  visible(ricgt_5,78). happensAt(walking(ricgt_5),78).
orientation(ricgt_3,11,79). coords(ricgt_3,239,229,79).  visible(ricgt_3,79). happensAt(walking(ricgt_3),79). orientation(ricgt_5,0,79). coords(ricgt_5,28,124,79).  visible(ricgt_5,79). happensAt(walking(ricgt_5),79).
orientation(ricgt_3,14,80). coords(ricgt_3,237,227,80).  visible(ricgt_3,80). happensAt(walking(ricgt_3),80). orientation(ricgt_5,0,80). coords(ricgt_5,28,124,80).  visible(ricgt_5,80). happensAt(walking(ricgt_5),80).
orientation(ricgt_3,12,81). coords(ricgt_3,237,225,81).  visible(ricgt_3,81). happensAt(walking(ricgt_3),81). orientation(ricgt_5,0,81). coords(ricgt_5,28,124,81).  visible(ricgt_5,81). happensAt(walking(ricgt_5),81).
orientation(ricgt_3,10,82). coords(ricgt_3,236,224,82).  visible(ricgt_3,82). happensAt(walking(ricgt_3),82). orientation(ricgt_5,0,82). coords(ricgt_5,28,124,82).  visible(ricgt_5,82). happensAt(walking(ricgt_5),82).
orientation(ricgt_3,8,83). coords(ricgt_3,235,223,83).  visible(ricgt_3,83). happensAt(walking(ricgt_3),83). orientation(ricgt_5,0,83). coords(ricgt_5,27,125,83).  visible(ricgt_5,83). happensAt(walking(ricgt_5),83).
orientation(ricgt_3,8,84). coords(ricgt_3,234,224,84).  visible(ricgt_3,84). happensAt(walking(ricgt_3),84). orientation(ricgt_5,0,84). coords(ricgt_5,27,127,84).  visible(ricgt_5,84). happensAt(walking(ricgt_5),84).
orientation(ricgt_3,6,85). coords(ricgt_3,233,221,85).  visible(ricgt_3,85). happensAt(walking(ricgt_3),85). orientation(ricgt_5,0,85). coords(ricgt_5,27,126,85).  visible(ricgt_5,85). happensAt(walking(ricgt_5),85).
orientation(ricgt_3,4,86). coords(ricgt_3,232,220,86).  visible(ricgt_3,86). happensAt(walking(ricgt_3),86). orientation(ricgt_5,0,86). coords(ricgt_5,27,126,86).  visible(ricgt_5,86). happensAt(walking(ricgt_5),86).
orientation(ricgt_3,2,87). coords(ricgt_3,231,219,87).  visible(ricgt_3,87). happensAt(walking(ricgt_3),87). orientation(ricgt_5,0,87). coords(ricgt_5,27,127,87).  visible(ricgt_5,87). happensAt(walking(ricgt_5),87).
orientation(ricgt_3,172,88). coords(ricgt_3,229,218,88).  visible(ricgt_3,88). happensAt(walking(ricgt_3),88). orientation(ricgt_5,0,88). coords(ricgt_5,27,127,88).  visible(ricgt_5,88). happensAt(walking(ricgt_5),88).
orientation(ricgt_3,168,89). coords(ricgt_3,226,216,89).  visible(ricgt_3,89). happensAt(walking(ricgt_3),89). orientation(ricgt_5,0,89). coords(ricgt_5,27,127,89).  visible(ricgt_5,89). happensAt(walking(ricgt_5),89).
orientation(ricgt_3,161,90). coords(ricgt_3,221,214,90).  visible(ricgt_3,90). happensAt(walking(ricgt_3),90). orientation(ricgt_5,0,90). coords(ricgt_5,27,127,90).  visible(ricgt_5,90). happensAt(walking(ricgt_5),90).
orientation(ricgt_3,158,91). coords(ricgt_3,219,213,91).  visible(ricgt_3,91). happensAt(walking(ricgt_3),91). orientation(ricgt_5,0,91). coords(ricgt_5,27,127,91).  visible(ricgt_5,91). happensAt(walking(ricgt_5),91).
orientation(ricgt_3,157,92). coords(ricgt_3,218,212,92).  visible(ricgt_3,92). happensAt(walking(ricgt_3),92). orientation(ricgt_5,0,92). coords(ricgt_5,27,126,92).  visible(ricgt_5,92). happensAt(walking(ricgt_5),92).
orientation(ricgt_3,157,93). coords(ricgt_3,217,210,93).  visible(ricgt_3,93). happensAt(walking(ricgt_3),93). orientation(ricgt_5,0,93). coords(ricgt_5,27,126,93).  visible(ricgt_5,93). happensAt(walking(ricgt_5),93).
orientation(ricgt_3,154,94). coords(ricgt_3,216,208,94).  visible(ricgt_3,94). happensAt(walking(ricgt_3),94). orientation(ricgt_5,0,94). coords(ricgt_5,27,126,94).  visible(ricgt_5,94). happensAt(walking(ricgt_5),94).
orientation(ricgt_3,153,95). coords(ricgt_3,215,205,95).  visible(ricgt_3,95). happensAt(walking(ricgt_3),95). orientation(ricgt_5,0,95). coords(ricgt_5,27,126,95).  visible(ricgt_5,95). happensAt(walking(ricgt_5),95).
orientation(ricgt_3,153,96). coords(ricgt_3,213,203,96).  visible(ricgt_3,96). happensAt(walking(ricgt_3),96). orientation(ricgt_5,0,96). coords(ricgt_5,27,126,96).  visible(ricgt_5,96). happensAt(walking(ricgt_5),96).
orientation(ricgt_3,152,97). coords(ricgt_3,212,199,97).  visible(ricgt_3,97). happensAt(walking(ricgt_3),97). orientation(ricgt_5,0,97). coords(ricgt_5,27,126,97).  visible(ricgt_5,97). happensAt(walking(ricgt_5),97).
orientation(ricgt_3,150,98). coords(ricgt_3,210,196,98).  visible(ricgt_3,98). happensAt(walking(ricgt_3),98). orientation(ricgt_5,0,98). coords(ricgt_5,27,126,98).  visible(ricgt_5,98). happensAt(walking(ricgt_5),98).
orientation(ricgt_3,155,99). coords(ricgt_3,212,190,99).  visible(ricgt_3,99). happensAt(walking(ricgt_3),99). orientation(ricgt_5,0,99). coords(ricgt_5,27,126,99).  visible(ricgt_5,99). happensAt(walking(ricgt_5),99).
#show holdsAt/2.