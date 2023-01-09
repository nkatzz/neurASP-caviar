
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

	orientation(lb1gt_0,156,36). coords(lb1gt_0,177,129,36).  visible(lb1gt_0,36). happensAt(walking(lb1gt_0),36). orientation(lb1gt_1,147,36). coords(lb1gt_1,72,88,36).  visible(lb1gt_1,36). happensAt(walking(lb1gt_1),36).
orientation(lb1gt_0,150,37). coords(lb1gt_0,178,128,37).  visible(lb1gt_0,37). happensAt(walking(lb1gt_0),37). orientation(lb1gt_1,147,37). coords(lb1gt_1,73,89,37).  visible(lb1gt_1,37). happensAt(walking(lb1gt_1),37).
orientation(lb1gt_0,150,38). coords(lb1gt_0,178,128,38).  visible(lb1gt_0,38). happensAt(walking(lb1gt_0),38). orientation(lb1gt_1,147,38). coords(lb1gt_1,73,89,38).  visible(lb1gt_1,38). happensAt(walking(lb1gt_1),38).
orientation(lb1gt_0,150,39). coords(lb1gt_0,179,127,39).  visible(lb1gt_0,39). happensAt(walking(lb1gt_0),39). orientation(lb1gt_1,147,39). coords(lb1gt_1,73,90,39).  visible(lb1gt_1,39). happensAt(walking(lb1gt_1),39).
orientation(lb1gt_0,150,40). coords(lb1gt_0,180,126,40).  visible(lb1gt_0,40). happensAt(walking(lb1gt_0),40). orientation(lb1gt_1,147,40). coords(lb1gt_1,73,90,40).  visible(lb1gt_1,40). happensAt(walking(lb1gt_1),40).
orientation(lb1gt_0,150,41). coords(lb1gt_0,180,126,41).  visible(lb1gt_0,41). happensAt(walking(lb1gt_0),41). orientation(lb1gt_1,147,41). coords(lb1gt_1,73,91,41).  visible(lb1gt_1,41). happensAt(walking(lb1gt_1),41).
orientation(lb1gt_0,150,42). coords(lb1gt_0,181,124,42).  visible(lb1gt_0,42). happensAt(walking(lb1gt_0),42). orientation(lb1gt_1,147,42). coords(lb1gt_1,73,91,42).  visible(lb1gt_1,42). happensAt(walking(lb1gt_1),42).
orientation(lb1gt_0,150,43). coords(lb1gt_0,182,123,43).  visible(lb1gt_0,43). happensAt(walking(lb1gt_0),43). orientation(lb1gt_1,147,43). coords(lb1gt_1,73,91,43).  visible(lb1gt_1,43). happensAt(walking(lb1gt_1),43).
orientation(lb1gt_0,150,44). coords(lb1gt_0,182,122,44).  visible(lb1gt_0,44). happensAt(walking(lb1gt_0),44). orientation(lb1gt_1,147,44). coords(lb1gt_1,73,91,44).  visible(lb1gt_1,44). happensAt(walking(lb1gt_1),44).
orientation(lb1gt_0,148,45). coords(lb1gt_0,182,121,45).  visible(lb1gt_0,45). happensAt(walking(lb1gt_0),45). orientation(lb1gt_1,147,45). coords(lb1gt_1,73,91,45).  visible(lb1gt_1,45). happensAt(walking(lb1gt_1),45).
orientation(lb1gt_0,148,46). coords(lb1gt_0,182,121,46).  visible(lb1gt_0,46). happensAt(walking(lb1gt_0),46). orientation(lb1gt_1,147,46). coords(lb1gt_1,73,92,46).  visible(lb1gt_1,46). happensAt(walking(lb1gt_1),46).
orientation(lb1gt_0,148,47). coords(lb1gt_0,182,120,47).  visible(lb1gt_0,47). happensAt(walking(lb1gt_0),47). orientation(lb1gt_1,147,47). coords(lb1gt_1,73,92,47).  visible(lb1gt_1,47). happensAt(walking(lb1gt_1),47).
orientation(lb1gt_0,148,48). coords(lb1gt_0,182,119,48).  visible(lb1gt_0,48). happensAt(walking(lb1gt_0),48). orientation(lb1gt_1,147,48). coords(lb1gt_1,73,92,48).  visible(lb1gt_1,48). happensAt(walking(lb1gt_1),48).
orientation(lb1gt_0,148,49). coords(lb1gt_0,182,119,49).  visible(lb1gt_0,49). happensAt(walking(lb1gt_0),49). orientation(lb1gt_1,147,49). coords(lb1gt_1,73,92,49).  visible(lb1gt_1,49). happensAt(walking(lb1gt_1),49).
orientation(lb1gt_0,148,50). coords(lb1gt_0,182,119,50).  visible(lb1gt_0,50). happensAt(walking(lb1gt_0),50). orientation(lb1gt_1,147,50). coords(lb1gt_1,73,92,50).  visible(lb1gt_1,50). happensAt(walking(lb1gt_1),50).
orientation(lb1gt_0,148,51). coords(lb1gt_0,182,119,51).  visible(lb1gt_0,51). happensAt(walking(lb1gt_0),51). orientation(lb1gt_1,147,51). coords(lb1gt_1,73,93,51).  visible(lb1gt_1,51). happensAt(walking(lb1gt_1),51).
orientation(lb1gt_0,148,52). coords(lb1gt_0,182,119,52).  visible(lb1gt_0,52). happensAt(walking(lb1gt_0),52). orientation(lb1gt_1,147,52). coords(lb1gt_1,73,94,52).  visible(lb1gt_1,52). happensAt(walking(lb1gt_1),52).
orientation(lb1gt_0,148,53). coords(lb1gt_0,182,119,53).  visible(lb1gt_0,53). happensAt(walking(lb1gt_0),53). orientation(lb1gt_1,147,53). coords(lb1gt_1,73,95,53).  visible(lb1gt_1,53). happensAt(walking(lb1gt_1),53).
orientation(lb1gt_0,148,54). coords(lb1gt_0,182,119,54).  visible(lb1gt_0,54). happensAt(walking(lb1gt_0),54). orientation(lb1gt_1,147,54). coords(lb1gt_1,73,97,54).  visible(lb1gt_1,54). happensAt(walking(lb1gt_1),54).
orientation(lb1gt_0,148,55). coords(lb1gt_0,182,119,55).  visible(lb1gt_0,55). happensAt(walking(lb1gt_0),55). orientation(lb1gt_1,147,55). coords(lb1gt_1,73,98,55).  visible(lb1gt_1,55). happensAt(walking(lb1gt_1),55).
orientation(lb1gt_0,148,56). coords(lb1gt_0,181,119,56).  visible(lb1gt_0,56). happensAt(walking(lb1gt_0),56). orientation(lb1gt_1,147,56). coords(lb1gt_1,73,99,56).  visible(lb1gt_1,56). happensAt(walking(lb1gt_1),56).
orientation(lb1gt_0,148,57). coords(lb1gt_0,181,119,57).  visible(lb1gt_0,57). happensAt(active(lb1gt_0),57). orientation(lb1gt_1,147,57). coords(lb1gt_1,73,100,57).  visible(lb1gt_1,57). happensAt(walking(lb1gt_1),57).
orientation(lb1gt_0,148,58). coords(lb1gt_0,181,119,58).  visible(lb1gt_0,58). happensAt(active(lb1gt_0),58). orientation(lb1gt_1,147,58). coords(lb1gt_1,73,100,58).  visible(lb1gt_1,58). happensAt(walking(lb1gt_1),58).
orientation(lb1gt_0,148,59). coords(lb1gt_0,181,119,59).  visible(lb1gt_0,59). happensAt(active(lb1gt_0),59). orientation(lb1gt_1,147,59). coords(lb1gt_1,73,100,59).  visible(lb1gt_1,59). happensAt(walking(lb1gt_1),59).
orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60).
#show holdsAt/2.